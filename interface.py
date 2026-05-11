import streamlit as st

st.set_page_config(page_title="JLPT Vocabulary Assistant", page_icon="🇯🇵")
st.title("JLPT Vocabulary Assistant")
st.caption("Ask questions about Japanese vocabulary from the JLPT dataset.")


@st.cache_resource(show_spinner="Loading RAG system...")
def load_rag_graph():
    """Load the compiled LangGraph RAG workflow once per Streamlit session."""
    from RAG import graph

    return graph


def ask_rag(question: str) -> str:
    """Run one question through the RAG graph and return the final answer text."""
    graph = load_rag_graph()
    result = graph.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input(
    "Ask about a Japanese word, meaning, reading, or JLPT level..."
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching vocabulary data..."):
            try:
                response = ask_rag(prompt)
            except Exception as exc:
                response = (
                    f"Sorry, I couldn't get an answer from the RAG system.\n\n`{exc}`"
                )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
