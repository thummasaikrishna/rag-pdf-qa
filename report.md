# Short Report: RAG-Based PDF Question Answering System

## 1. Architecture

The system follows the standard Retrieval-Augmented Generation pattern,
split into an **indexing phase** (run once per document set) and a
**query phase** (run per user question):

**Indexing phase**
1. **PDF extraction** — `pypdf` reads each uploaded file page by page,
   keeping the page number attached to its text so later answers can cite
   an exact source page.
2. **Chunking** — page text is normalized (whitespace collapsed) and split
   into overlapping windows. The splitter first tries to land the chunk
   boundary on a sentence end within the last 20% of the window, only
   falling back to a hard character cut if no sentence boundary is nearby.
   Overlap (default 150 characters) is repeated at the start of the next
   chunk so a fact split across a chunk boundary is not lost.
3. **Embedding** — each chunk is encoded into a 384-dimensional vector with
   a Sentence-Transformers model (`all-MiniLM-L6-v2` by default). Embeddings
   are L2-normalized so inner product equals cosine similarity.
4. **Vector storage** — normalized embeddings go into a FAISS
   `IndexFlatIP` (exact search, fine at this scale). Chunk metadata
   (source filename, page, chunk id) is kept in a parallel Python list.
   A BM25 index (`rank-bm25`) is built over the same chunks for optional
   hybrid search.

**Query phase**
5. **Retrieval** — the question is embedded with the same model and
   compared against the FAISS index for the top-k most similar chunks. If
   hybrid mode is enabled, BM25 keyword scores and vector similarity scores
   are min-max normalized and blended with an adjustable weight (`alpha`),
   which helps on queries containing exact terms (product codes, names)
   that pure semantic search can under-rank.
6. **Answer generation** — retrieved chunks are formatted into a single
   context block with inline source tags (`[Source N: file, page P]`) and
   inserted into a strict prompt instructing the LLM (via the Groq API,
   Llama 3.3 70B by default) to answer *only* from that context, and to
   say explicitly when the answer isn't present. Recent chat turns are
   also passed in so follow-up questions ("what about the international
   rate?") resolve correctly.
7. **Presentation** — the Streamlit sidebar only has PDF upload, process, and
   clear chat. The main page shows the answer and an expandable Sources
   section with file name, page, score, and keyword highlighting.

## 2. Libraries Used

| Library | Purpose |
|---|---|
| `streamlit` | Web UI, file upload, chat interface, session state |
| `pypdf` | PDF parsing / text extraction |
| `sentence-transformers` | Local, free embedding generation |
| `faiss-cpu` | Vector similarity search |
| `rank-bm25` | Keyword-based scoring for hybrid search |
| `groq` | Chat-completions client for the LLM (Llama 3.3 / 3.1, Gemma2) |
| `reportlab` | Generates the two bundled sample PDFs |

The RAG logic was implemented directly on these primitives rather than via
LangChain/LlamaIndex, so every step (chunk boundaries, score blending,
prompt construction) is explicit and easy to reason about or swap out —
at the cost of losing some of the pre-built connectors those frameworks
provide.

## 3. Challenges Faced

- **Chunk boundaries cutting mid-sentence**: a naive fixed-length split
  regularly broke sentences (and sometimes numbers/units) in half, which
  hurt both embedding quality and the LLM's ability to quote a clean fact.
  Solved by snapping to the nearest sentence-ending punctuation within the
  tail of each window before falling back to a hard cut.
- **Comparing vector and keyword scores**: FAISS cosine similarity and raw
  BM25 scores live on completely different scales, so a naive sum favors
  whichever happens to be numerically larger. Both are min-max normalized
  into a comparable range before blending in hybrid mode.
- **Scanned / image-only PDFs**: `pypdf` extracts embedded text but returns
  nothing for scanned pages with no text layer, silently producing empty
  chunks. The extractor now skips blank pages and reports how many pages
  yielded text, so this failure mode is visible instead of silent.
- **Grounding vs. refusal trade-off**: an early version of the prompt still
  let the model answer from general knowledge when retrieval was weak.
  Tightening the prompt to explicitly require the "I couldn't find this
  information..." fallback reduced hallucinated answers on out-of-scope
  questions.
- **Testing without burning API calls**: PDF extraction and chunking were
  checked against the bundled sample PDFs with a small offline test file.
  Embedding download and Groq generation need internet plus an API key, so
  I verified those by running the Streamlit app locally after indexing
  the sample documents.

## 4. Future Improvements

- **Persistent storage**: currently the FAISS index lives only in Streamlit
  session state; `VectorStore.save()`/`.load()` already exist in
  `rag_core.py` but aren't yet wired into the UI for reusing an index
  across app restarts.
- **Document summaries**: a one-click "Summarize this document" button
  using the same LLM, independent of the Q&A flow.
- **Voice input**: browser-based speech-to-text (e.g. the Web Speech API
  via a small custom Streamlit component) to populate the chat box.
- **Re-ranking**: add a cross-encoder re-ranking pass over the top ~20
  vector hits before truncating to top-k, which typically improves
  retrieval precision more than tuning chunk size alone.
- **Cloud vector DB**: swap FAISS for Pinecone or a hosted ChromaDB
  instance to support larger corpora and multi-user persistence.
- **Evaluation harness**: a small labeled QA set per sample PDF to measure
  retrieval accuracy (recall@k) and answer faithfulness automatically,
  rather than relying on manual spot-checks.
