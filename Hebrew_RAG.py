import os
from typing import Literal

import dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.document_loaders import CSVLoader
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# 1. Preprocess documents — load Hebrew vocabulary data
# ---------------------------------------------------------------------------
# Expected CSV columns:
#   word            – the Hebrew word (e.g. לִכְתֹּב)
#   transliteration – Latin transliteration (e.g. lichtov)
#   meaning         – English meaning (e.g. to write)
#   part_of_speech  – noun / verb / adjective / ...
#   root            – the shoresh / triliteral root (e.g. כ-ת-ב)
#   binyan          – for verbs only: Pa'al, Nif'al, Pi'el, Pu'al,
#                     Hif'il, Huf'al, Hitpa'el  (empty for non-verbs)
#   cefr_level      – A1 / A2 / B1 / B2 / C1 / C2  (UNICert grading)

document_paths = [
    os.path.join(os.path.dirname(__file__), "documents", "hebrew_vocab.csv")
]

docs = [CSVLoader(path).load() for path in document_paths]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# ---------------------------------------------------------------------------
# 2. Create a retriever tool
# ---------------------------------------------------------------------------

vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=OllamaEmbeddings(model="embeddinggemma"),
)
retriever = vectorstore.as_retriever()


@tool
def retrieve_hebrew_info(query: str) -> str:
    """Search and return Hebrew vocabulary entries matching the query.
    Each entry includes the word, transliteration, meaning, part of speech,
    root (shoresh), binyan (for verbs), and CEFR/UNICert level.
    """
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


retriever_tool = retrieve_hebrew_info


def retrieve_node(state: MessagesState):
    """Custom tool node that tolerates malformed tool arguments.

    qwen2.5:3b occasionally emits the query as a wrapped schema object
    like {'type': 'string', 'value': 'X'} instead of the plain string 'X'.
    This node unwraps that case before invoking the retriever.
    """
    last_message = state["messages"][-1]
    tool_messages = []
    for tool_call in last_message.tool_calls:
        raw_query = tool_call["args"].get("query", "")
        if isinstance(raw_query, dict):
            raw_query = raw_query.get("value") or raw_query.get("query") or ""
        if not isinstance(raw_query, str):
            raw_query = str(raw_query)
        result = retriever_tool.invoke({"query": raw_query})
        tool_messages.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )
    return {"messages": tool_messages}

# ---------------------------------------------------------------------------
# 3. Generate query — LLM decides whether to retrieve or respond directly
# ---------------------------------------------------------------------------

response_model = init_chat_model(
    "qwen2.5:3b", model_provider="ollama", temperature=0
)


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state.
    Given the question, it will decide to retrieve using the retriever tool,
    or simply respond to the user.
    """
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# 4. Grade documents — check if retrieved content is relevant
# ---------------------------------------------------------------------------

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question.\n"
    "Here is the retrieved document:\n\n{context}\n\n"
    "Here is the user question: {question}\n"
    "If the document contains keyword(s) or semantic meaning related to the user question, "
    "grade it as relevant.\n"
    "Respond with exactly one word: yes or no."
)


grader_model = init_chat_model(
    "qwen2.5:3b", model_provider="ollama", temperature=0
)


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader_model.invoke([{"role": "user", "content": prompt}])
    score = response.content.strip().lower()

    if score.startswith("yes"):
        return "generate_answer"
    else:
        return "rewrite_question"


# ---------------------------------------------------------------------------
# 5. Rewrite question — improve query when retrieval is irrelevant
# ---------------------------------------------------------------------------

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}


# ---------------------------------------------------------------------------
# 6. Generate answer — final answer from question + retrieved context
# ---------------------------------------------------------------------------

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks about Hebrew vocabulary.\n"
    "Use the following pieces of retrieved context to answer the question.\n"
    "When the information is available in the context, always include:\n"
    "  • the meaning of the word\n"
    "  • the root (shoresh), e.g. כ-ת-ב\n"
    "  • for verbs: the binyan "
    "(Pa'al/Qal, Nif'al, Pi'el, Pu'al, Hif'il, Huf'al, Hitpa'el)\n"
    "  • the CEFR / UNICert level (A1–C2)\n"
    "If you don't know the answer, just say that you don't know. "
    "Use three to four sentences maximum and keep the answer concise.\n"
    "Question: {question}\n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# 7. Assemble the graph
# ---------------------------------------------------------------------------

workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the retrieval node is called
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()


# ---------------------------------------------------------------------------
# 8. Run the agentic RAG
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    question = (
        "What is the meaning of the Hebrew verb לִכְתֹּב, what is its root (shoresh) "
        "and binyan, and what CEFR/UNICert level is it?"
    )
    print(f"Question: {question}\n")

    for chunk in graph.stream(
        {
            "messages": [
                {"role": "user", "content": question},
            ]
        },
        config={"recursion_limit": 10},
    ):
        for node, update in chunk.items():
            print(f"--- Update from node: {node} ---")
            update["messages"][-1].pretty_print()
            print()