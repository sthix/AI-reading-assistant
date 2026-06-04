import streamlit as st

LANGUAGE_OPTIONS = {
    "Japanese": {
        "flag": "🇯🇵",
        "title": "JLPT Vocabulary Assistant",
        "caption": "Ask questions about Japanese vocabulary from the JLPT dataset.",
        "scale_label": "JLPT level (N5–N1)",
        "input_placeholder": (
            "Ask about a Japanese word, meaning, reading, or JLPT level..."
        ),
        "text_label": "Japanese text",
        "text_placeholder": "Paste your Japanese text here, or upload a text file above...",
        "spinner": "Searching vocabulary data...",
        "rag_graph_attr": "graph_japanese",
    },
    "Hebrew": {
        "flag": "🇮🇱",
        "title": "Hebrew Vocabulary Assistant",
        "caption": "Ask questions about Hebrew vocabulary from the CEFR dataset.",
        "scale_label": "CEFR level (A1–C1)",
        "input_placeholder": (
            "Ask about a Hebrew word, meaning, or CEFR level..."
        ),
        "text_label": "Hebrew text",
        "text_placeholder": "Paste your Hebrew text here, or upload a text file above...",
        "spinner": "Searching vocabulary data...",
        "rag_graph_attr": "graph_hebrew",
    },
}


def _load_graph(language: str):
    """Return the compiled RAG graph for the selected language (cached per session)."""
    from RAG import graph_hebrew, graph_japanese

    graphs = {"Japanese": graph_japanese, "Hebrew": graph_hebrew}

    @st.cache_resource(show_spinner=f"Loading {language} RAG system...")
    def _cached():
        return graphs[language]

    return _cached()


def ask_rag(question: str, language: str) -> str:
    """Run one question through the RAG graph and return the final answer text."""
    graph = _load_graph(language)
    result = graph.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# Sidebar — language mode selector
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    language = st.radio(
        "Language mode",
        options=list(LANGUAGE_OPTIONS.keys()),
        index=0,
        help="Switch between the Japanese (JLPT) and Hebrew (CEFR) tutor.",
    )

# ---------------------------------------------------------------------------
# Page config (must run before any st.* UI elements other than the sidebar)
# ---------------------------------------------------------------------------

config = LANGUAGE_OPTIONS[language]

st.set_page_config(
    page_title=config["title"], page_icon=config["flag"], layout="wide"
)


title_col1, title_col2 = st.columns([2, 1])
with title_col1:
    st.title(config["title"])
    st.caption(config["caption"])

with title_col2:
    uploaded_file = st.file_uploader(
        label="Text upload",
        type=["txt", "md", "csv"],
        help="Upload a UTF-8 text file to place its content into the text field.",
    )
    text_key = f"{language.lower()}_text"
    file_name_key = f"{language.lower()}_uploaded_file_name"
    if uploaded_file is not None:
        uploaded_text = uploaded_file.getvalue().decode("utf-8")
        if st.session_state.get(file_name_key) != uploaded_file.name:
            st.session_state[text_key] = uploaded_text
            st.session_state[file_name_key] = uploaded_file.name

    if st.button("Clear text"):
        st.session_state[text_key] = ""
        st.session_state[file_name_key] = None
        st.rerun()

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.text("Your AI tutor")
    st.space(12)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input(config["input_placeholder"]):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(config["spinner"]):
                try:
                    response = ask_rag(prompt, language)
                except Exception as exc:
                    response = (
                        f"Sorry, I couldn't get an answer from the RAG system.\n\n`{exc}`"
                    )
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    st.text("Text")
    target_text = st.text_area(
        label=config["text_label"],
        key=text_key,
        height=350,
        placeholder=config["text_placeholder"],
        label_visibility="hidden",
    )

text_length = len(target_text)

with col3:
    st.text("Statistics")
    st.space(12)
    st.metric(label="Text length", value=text_length, border=True)
