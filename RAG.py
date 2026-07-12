import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

dotenv.load_dotenv()

Language = Literal["japanese", "hebrew"]
Provider = Literal["ollama", "groq"]

BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"

LANGUAGE_CONFIG: dict[str, dict[str, object]] = {
    "japanese": {
        "label": "Japanese",
        "document_paths": [
            DOCUMENTS_DIR / "JLPT_vocab_ALL.csv",
            DOCUMENTS_DIR / "JLPT_kanji_ALL.csv",
        ],
        "persist_directory": BASE_DIR / "chroma_db_japanese",
        "tool_name": "retrieve_japanese_info",
        "tool_description": (
            "Search Japanese JLPT vocabulary and kanji entries by word, reading, "
            "English meaning, or JLPT level. Rows include Japanese form, reading, "
            "meaning, and JLPT difficulty N5-N1."
        ),
        "document_kinds": "Japanese vocabulary, readings, meanings, kanji, and JLPT N5-N1 levels",
        "mentor_prompt": (
            "You are a Japanese vocabulary mentor. Use the retrieved JLPT rows to "
            "answer the learner's question. If relevant, include the Japanese word, "
            "reading, concise English meaning, and JLPT difficulty level N5-N1."
        ),
        "rewrite_prompt": "Formulate an improved Japanese JLPT vocabulary search question:",
    },
    "hebrew": {
        "label": "Hebrew",
        "document_paths": [DOCUMENTS_DIR / "hebrew_cefr_final.csv"],
        "persist_directory": BASE_DIR / "chroma_db_hebrew_final",
        "tool_name": "retrieve_hebrew_info",
        "tool_description": (
            "Search Hebrew vocabulary entries by Hebrew word or English meaning. "
            "Rows come from hebrew_cefr_final.csv and include rank, Hebrew word, "
            "English gloss, CEFR level A1-C1, confidence, and source."
        ),
        "document_kinds": "Hebrew vocabulary, English glosses, and CEFR A1-C1 levels",
        "mentor_prompt": (
            "You are a Hebrew vocabulary mentor. Use the retrieved Hebrew CEFR list "
            "rows to answer the learner's question. If relevant, include the Hebrew "
            "word, concise English gloss, and CEFR difficulty level A1-C1."
        ),
        "rewrite_prompt": "Formulate an improved Hebrew CEFR vocabulary search question:",
    },
}

# Embeddings always stay on Ollama: the persisted Chroma stores were built with
# embeddinggemma vectors, so swapping the chat provider must not touch retrieval.
embedding_model = OllamaEmbeddings(model="embeddinggemma")

PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "ollama": {"model": "gemma4:31b-cloud", "model_provider": "ollama"},
    "groq": {
        "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "model_provider": "groq",
    },
}


def load_csv_documents(language: Language):
    config = LANGUAGE_CONFIG[language]
    paths = config["document_paths"]
    docs = []
    for path in paths:  # type: ignore[assignment]
        document_path = Path(path)
        if not document_path.exists():
            raise FileNotFoundError(
                f"Missing {config['label']} RAG document: {document_path}"
            )
        docs.extend(CSVLoader(document_path).load())
    return docs


@lru_cache(maxsize=2)
def get_retriever(language: Language):
    docs_list = load_csv_documents(language)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    persist_directory = str(LANGUAGE_CONFIG[language]["persist_directory"])
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )
    # A leftover chroma.sqlite3 without vector segments used to pass a
    # directory-not-empty check while the collection held zero documents,
    # so count the stored embeddings instead of listing files.
    if vectorstore._collection.count() == 0:
        # Ollama's embedding runner dies on a single oversized embed request,
        # so the full dataset has to be inserted in batches.
        batch_size = 200
        for start in range(0, len(doc_splits), batch_size):
            vectorstore.add_documents(doc_splits[start : start + batch_size])

    return vectorstore.as_retriever(search_kwargs={"k": 6})


def build_mentor_prompt(language: Language, question: str, context: str) -> str:
    config = LANGUAGE_CONFIG[language]
    return (
        f"{config['mentor_prompt']} "
        "If you don't know the answer from the context, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n"
        f"Question: {question}\n"
        f"Context: {context}"
    )


@lru_cache(maxsize=4)
def build_graph(language: Language, provider: Provider = "ollama"):
    """Compile a LangGraph RAG workflow for Japanese or Hebrew."""
    config = LANGUAGE_CONFIG[language]
    retriever = get_retriever(language)
    model_config = PROVIDER_MODELS[provider]

    @tool
    def retrieve_language_info(query: str) -> str:
        """Search the selected language vocabulary dataset."""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    retrieve_language_info.name = str(config["tool_name"])
    retrieve_language_info.description = str(config["tool_description"])
    retriever_tool = retrieve_language_info

    response_model = init_chat_model(temperature=0.2, **model_config)
    grader_model = init_chat_model(temperature=0, **model_config)

    def generate_query_or_respond(state: MessagesState):
        response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}

    grade_prompt = (
        "You are a grader assessing relevance of a retrieved vocabulary row "
        "to a user question.\n"
        "Here is the retrieved document:\n\n{context}\n\n"
        "Here is the user question: {question}\n"
        f"If the document contains information related to {config['document_kinds']} "
        "and the user question, grade it as relevant.\n"
        "Respond with exactly one word: yes or no."
    )

    def grade_documents(
        state: MessagesState,
    ) -> Literal["generate_answer", "rewrite_question"]:
        question = state["messages"][0].content
        context = state["messages"][-1].content
        # Each rewrite appends a HumanMessage after the original question.
        # Allow a single rewrite, then answer with the best context we have:
        # every extra loop costs 3-4 sequential LLM calls, and an unbounded
        # loop kept chat requests spinning until the recursion limit.
        rewrite_count = sum(
            isinstance(message, HumanMessage) for message in state["messages"]
        ) - 1
        if rewrite_count >= 1:
            return "generate_answer"
        prompt = grade_prompt.format(question=question, context=context)
        response = grader_model.invoke([{"role": "user", "content": prompt}])
        score = response.content.strip().lower()
        if score.startswith("yes"):
            return "generate_answer"
        return "rewrite_question"

    rewrite_prompt = (
        "Look at the input and infer the intended vocabulary lookup.\n"
        "Here is the initial question:"
        "\n ------- \n"
        "{question}"
        "\n ------- \n"
        f"{config['rewrite_prompt']}\n"
        "Return only the single improved question, with no explanations, "
        "options, or extra text."
    )

    def rewrite_question(state: MessagesState):
        question = state["messages"][0].content
        prompt = rewrite_prompt.format(question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=response.content)]}

    def generate_answer(state: MessagesState):
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = build_mentor_prompt(language, question, context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END},
    )
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()


graph_japanese = build_graph("japanese")
graph_hebrew = build_graph("hebrew")
graph = graph_japanese


if __name__ == "__main__":
    question = "What is the meaning of שלום, and what is its CEFR level?"
    print(f"Question: {question}\n")
    for chunk in graph_hebrew.stream(
        {"messages": [{"role": "user", "content": question}]}
    ):
        for node, update in chunk.items():
            print(f"--- Update from node: {node} ---")
            update["messages"][-1].pretty_print()
            print()
