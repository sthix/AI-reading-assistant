import streamlit as st

st.set_page_config(
    page_title="JLPT Vocabulary Assistant", page_icon="🇯🇵", layout="wide"
)


title_col1, title_col2 = st.columns([2, 1])
with title_col1:
    st.title("JLPT Vocabulary Assistant")
    st.caption("Ask questions about Japanese vocabulary from the JLPT dataset.")

with title_col2:
    uploaded_file = st.file_uploader(
        label="Text upload",
        type=["txt", "md", "csv"],
        help="Upload a UTF-8 text file to place its content into the text field.",
    )
    if uploaded_file is not None:
        uploaded_text = uploaded_file.getvalue().decode("utf-8")
        if st.session_state.get("uploaded_file_name") != uploaded_file.name:
            st.session_state.japanese_text = uploaded_text
            st.session_state.uploaded_file_name = uploaded_file.name

    if st.button("Clear text"):
        st.session_state.japanese_text = ""
        st.session_state.uploaded_file_name = None
        st.rerun()

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.text("Your AI tutor")
    st.space(12)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

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
                    response = f"Sorry, I couldn't get an answer from the RAG system.\n\n`{exc}`"
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

label = "Japanese text"

with col2:
    st.text("Text")
    jap_text = st.text_area(
        label=label,
        key="japanese_text",
        height=350,
        placeholder="Paste your Japanese text here, or upload a text file above...",
        label_visibility="hidden",
    )

text_length = len(jap_text)

with col3:
    st.text("Statistics")
    st.space(12)
    st.metric(label="Text length", value=text_length, border=True)
