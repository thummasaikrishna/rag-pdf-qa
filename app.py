"""
app.py

Streamlit UI. Run with:

    streamlit run app.py
"""

import os
import tempfile

import streamlit as st

from rag_core import (
    extract_pages_from_pdf,
    build_chunks_for_documents,
    VectorStore,
    GroqLLM,
    highlight_keywords,
)

st.set_page_config(page_title="PDF Q&A (RAG)", page_icon="📄", layout="wide")


def _load_local_env():
    """Read GROQ_API_KEY from a .env file in this folder if present."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _get_groq_api_key():
    """Read Groq key from .env / environment, or from Streamlit secrets."""
    _load_local_env()
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        return ""

# --------------------------------------------------------------------------
# Session state initialisation
# --------------------------------------------------------------------------
defaults = {
    "vector_store": None,
    "chat_history": [],       # list of {"role": "user"/"assistant", "content": str}
    "processed_files": [],    # names of files already indexed
    "llm": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# chunking / retrieval / model values
chunk_size = 800
chunk_overlap = 150
top_k = 4
use_hybrid = False
alpha = 0.5
embedding_model_name = "all-MiniLM-L6-v2"
groq_model = "llama-3.3-70b-versatile"
api_key_input = _get_groq_api_key()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
    )

    process_clicked = st.button("🚀 Process documents", type="primary", use_container_width=True)

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# --------------------------------------------------------------------------
# Main title
# --------------------------------------------------------------------------
st.title("📄 RAG-Based PDF Question Answering System")
st.markdown(
    "<p style='color:red; font-weight:bold;'>"
    "(If Language Model is not working wait for 16 minutes and try again)"
    "</p>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Document processing pipeline (steps 1–4 of the workflow)
# --------------------------------------------------------------------------
def process_documents(files, chunk_size, chunk_overlap, embedding_model_name):
    documents = {}
    with st.status("Processing documents...", expanded=True) as status:
        for f in files:
            status.write(f"📥 Extracting text from **{f.name}**...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getbuffer())
                tmp_path = tmp.name
            try:
                pages = extract_pages_from_pdf(tmp_path)
            finally:
                os.unlink(tmp_path)
            documents[f.name] = pages
            status.write(f"   → {len(pages)} page(s) with extractable text")

        status.write("✂️ Splitting text into chunks...")
        chunks = build_chunks_for_documents(documents, chunk_size, chunk_overlap)
        status.write(f"   → {len(chunks)} chunk(s) created")

        status.write(f"🧠 Generating embeddings with `{embedding_model_name}`...")
        store = VectorStore(embedding_model_name)
        store.build(chunks)
        status.write("   → embeddings generated and indexed in FAISS")

        status.update(label="✅ Documents processed and indexed!", state="complete")

    return store


if process_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF first.")
    else:
        try:
            st.session_state.vector_store = process_documents(
                uploaded_files, chunk_size, chunk_overlap, embedding_model_name
            )
            st.session_state.processed_files = [f.name for f in uploaded_files]
            st.session_state.chat_history = []
        except Exception as e:
            st.error(f"Failed to process documents: {e}")


# --------------------------------------------------------------------------
# Status panel
# --------------------------------------------------------------------------
if st.session_state.processed_files:
    st.success(f"📚 Indexed documents: {', '.join(st.session_state.processed_files)}")


# --------------------------------------------------------------------------
# Chat interface (steps 5–7: retrieve, generate, display with sources)
# --------------------------------------------------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📌 Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['source']}**, page {src['page']} "
                        f"(relevance: {src['score']:.2f})"
                    )
                    st.markdown(f"> {src['highlighted_text']}")

question = st.chat_input("Ask a question about your uploaded documents...")

if question:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process at least one PDF before asking questions.")
    elif not api_key_input:
        st.warning("Groq API key is missing. Set GROQ_API_KEY and try again.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context..."):
                store = st.session_state.vector_store
                if use_hybrid:
                    retrieved = store.hybrid_search(question, k=top_k, alpha=alpha)
                else:
                    retrieved = store.search(question, k=top_k)

            with st.spinner("Generating answer..."):
                try:
                    llm = GroqLLM(api_key=api_key_input, model=groq_model)
                    # last 6 turns for lightweight conversation memory
                    history_for_llm = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history[-6:]
                        if m["role"] in ("user", "assistant")
                    ]
                    answer = llm.answer(question, retrieved, chat_history=history_for_llm)
                except Exception as e:
                    answer = f"⚠️ Error calling the LLM: {e}"

            st.markdown(answer)

            sources = []
            if retrieved:
                with st.expander("📌 Sources"):
                    for r in retrieved:
                        highlighted = highlight_keywords(r.chunk.text, question)
                        st.markdown(
                            f"**{r.chunk.source}**, page {r.chunk.page} "
                            f"(relevance: {r.score:.2f})"
                        )
                        st.markdown(f"> {highlighted}")
                        sources.append(
                            {
                                "source": r.chunk.source,
                                "page": r.chunk.page,
                                "score": r.score,
                                "highlighted_text": highlighted,
                            }
                        )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
