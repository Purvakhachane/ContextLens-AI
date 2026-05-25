from pathlib import Path

import streamlit as st

from src.config import settings
from src.rag_chain import answer_question, retrieve_with_scores
from src.vector_store import build_vector_store


st.set_page_config(page_title="ContextLens AI", page_icon="search", layout="wide")

st.title("ContextLens AI")
st.caption("Grounded answers with evidence confidence and source transparency")

if not settings.has_valid_provider_key:
    st.warning(
        "Add a real API key in your `.env` file before rebuilding the index or asking questions. "
        "After saving it, restart Streamlit."
    )

with st.sidebar:
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Add documents",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Save uploaded documents", use_container_width=True):
        settings.data_dir.mkdir(exist_ok=True)
        for uploaded_file in uploaded_files:
            safe_name = Path(uploaded_file.name).name
            target = settings.data_dir / safe_name
            target.write_bytes(uploaded_file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} document(s). Rebuild the index next.")

    if st.button("Rebuild index", use_container_width=True):
        if not settings.has_valid_provider_key:
            st.error("API key is missing or still uses the placeholder value.")
        else:
            try:
                with st.spinner("Indexing documents..."):
                    store = build_vector_store(reset=True)
                    st.success(f"Indexed {store._collection.count()} chunks.")
            except Exception as error:
                st.error(f"Indexing failed: {error}")

    st.divider()
    mode = st.radio("Mode", ["Answer with LLM", "Retrieve only"])
    style = st.selectbox("Answer style", ["Balanced", "Brief", "Detailed", "Executive"])

question = st.chat_input("Ask a question about your indexed documents")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            if not settings.has_valid_provider_key:
                st.error("API key is missing or invalid. Update `.env`, then restart Streamlit.")
            else:
                try:
                    if mode == "Retrieve only":
                        scored_docs = retrieve_with_scores(question)
                        st.subheader("Evidence Lens")
                        for index, (doc, score) in enumerate(scored_docs, start=1):
                            source = doc.metadata.get("source", "unknown source")
                            page = doc.metadata.get("page")
                            label = f"{source}, page {page + 1}" if page is not None else source
                            st.progress(max(0, min(1, score)), text=f"Source {index} relevance: {score:.0%}")
                            with st.expander(label, expanded=index == 1):
                                st.write(doc.page_content)
                    else:
                        result = answer_question(question, style=style)
                        score_col, guard_col = st.columns(2)
                        score_col.metric(
                            "Evidence Confidence",
                            result["confidence_label"],
                            f"{result['confidence']:.0%}",
                        )
                        guard_col.metric("Hallucination Guard", "Active")
                        st.write(result["answer"])
                        st.subheader("Evidence Lens")
                        for index, (doc, score) in enumerate(result["scored_sources"], start=1):
                            source = doc.metadata.get("source", "unknown source")
                            page = doc.metadata.get("page")
                            label = f"{source}, page {page + 1}" if page is not None else source
                            st.progress(max(0, min(1, score)), text=f"Source {index} relevance: {score:.0%}")
                            with st.expander(label):
                                st.write(doc.page_content)
                except Exception as error:
                    st.error(f"Request failed: {error}")
