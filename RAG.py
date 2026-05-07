import getpass
import os

import dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.document_loaders import CSVLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState

dotenv.load_dotenv()


def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")


_set_env("GROQ_API_KEY")


document_paths = [
    "/Users/sascha/neuefische_course/Language_Assistant/documents/jlpt_vocab.csv"
]

docs = [CSVLoader(doc).load() for doc in document_paths]

docs[0][0].page_content.strip()[:1000]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

doc_splits[0].page_content.strip()

vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=OllamaEmbeddings(model="embeddinggemma")
)
retriever = vectorstore.as_retriever()


@tool
def retrieve_jlpt_info(query: str) -> str:
    """Search and return information about JLPT vocabulary."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


retriever_tool = retrieve_jlpt_info

retriever_tool.invoke({"query": "JLPT vocabulary sorted by level"})


response_model = init_chat_model(
    "kimi-k2.6:cloud", model_provider="ollama", temperature=0
)


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


input = {"messages": [{"role": "user", "content": "Hi! Which JLPT level is 勉強?"}]}

generate_query_or_respond(input)["messages"][-1].pretty_print()
