"""
rag_core.py

RAG pipeline used by the Streamlit app:

1. Extract text from PDFs (keep page numbers)
2. Split into overlapping chunks
3. Embed chunks with Sentence-Transformers
4. Store / search with FAISS
5. Optional BM25 hybrid search
6. Call Groq to generate the answer from retrieved context
"""

from __future__ import annotations

import os
import re
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None


# --------------------------------------------------------------------------
# 1. Data structures
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single chunk of text with metadata used for citations."""
    text: str
    source: str          # original filename
    page: int             # 1-indexed page number
    chunk_id: int          # position within the whole corpus


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


# --------------------------------------------------------------------------
# 2. PDF extraction
# --------------------------------------------------------------------------

def extract_pages_from_pdf(file_path: str) -> List[Dict]:
    """
    Extracts text from a PDF, page by page.

    Returns a list of dicts: {"page": int, "text": str}
    Blank / unreadable pages are skipped.
    """
    if PdfReader is None:
        raise ImportError("pypdf is not installed. `pip install pypdf`.")

    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


# --------------------------------------------------------------------------
# 3. Chunking
# --------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[str]:
    """
    Splits `text` into overlapping chunks measured in characters.

    A simple, dependency-free recursive-ish splitter:
    - Tries to break on paragraph/sentence boundaries near the chunk_size
      limit so chunks don't cut words/sentences in half when possible.
    - `chunk_overlap` characters from the tail of one chunk are repeated
      at the start of the next chunk to preserve context across the split.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        if end < n:
            # try to end on a sentence boundary within the last 20% of the window
            window_start = start + int(chunk_size * 0.8)
            boundary = -1
            for match in re.finditer(r"[.!?]\s", text[window_start:end]):
                boundary = window_start + match.end()
            if boundary != -1:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def build_chunks_for_documents(
    documents: Dict[str, List[Dict]],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """
    documents: {filename: [{"page": int, "text": str}, ...]}
    Returns a flat list of Chunk objects across all documents.
    """
    all_chunks: List[Chunk] = []
    cid = 0
    for filename, pages in documents.items():
        for page_info in pages:
            pieces = chunk_text(page_info["text"], chunk_size, chunk_overlap)
            for piece in pieces:
                all_chunks.append(
                    Chunk(text=piece, source=filename, page=page_info["page"], chunk_id=cid)
                )
                cid += 1
    return all_chunks


# --------------------------------------------------------------------------
# 4. Embeddings + Vector store (FAISS) + optional BM25 hybrid search
# --------------------------------------------------------------------------

class VectorStore:
    """
    Wraps a Sentence-Transformers embedding model + a FAISS index.
    Also keeps a BM25 index over the same chunks for optional hybrid search.
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "`pip install sentence-transformers`."
            )
        if faiss is None:
            raise ImportError("faiss is not installed. `pip install faiss-cpu`.")

        self.embedding_model_name = embedding_model_name
        self.model = SentenceTransformer(embedding_model_name)
        self.index: Optional["faiss.Index"] = None
        self.chunks: List[Chunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []

    # ---- building ----
    def build(self, chunks: List[Chunk], batch_size: int = 64) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # so inner product == cosine similarity
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        if BM25Okapi is not None:
            self._tokenized_corpus = [self._tokenize(t) for t in texts]
            self.bm25 = BM25Okapi(self._tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    # ---- persistence ----
    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dir_path, "index.faiss"))
        with open(os.path.join(dir_path, "meta.pkl"), "wb") as f:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "embedding_model_name": self.embedding_model_name,
                    "tokenized_corpus": self._tokenized_corpus,
                },
                f,
            )

    @classmethod
    def load(cls, dir_path: str) -> "VectorStore":
        with open(os.path.join(dir_path, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        store = cls(meta["embedding_model_name"])
        store.index = faiss.read_index(os.path.join(dir_path, "index.faiss"))
        store.chunks = meta["chunks"]
        store._tokenized_corpus = meta.get("tokenized_corpus", [])
        if BM25Okapi is not None and store._tokenized_corpus:
            store.bm25 = BM25Okapi(store._tokenized_corpus)
        return store

    # ---- retrieval ----
    def search(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        """Pure vector (semantic) search."""
        if self.index is None:
            return []
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(q_emb, min(k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=float(score)))
        return results

    def hybrid_search(self, query: str, k: int = 4, alpha: float = 0.5) -> List[RetrievedChunk]:
        """
        Hybrid search: combines normalized vector similarity with normalized
        BM25 keyword scores. `alpha` weights vector score vs keyword score
        (alpha=1.0 -> pure vector, alpha=0.0 -> pure keyword).
        """
        if self.index is None:
            return []
        if self.bm25 is None:
            return self.search(query, k)

        # vector scores over the whole corpus
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        vec_scores, vec_idxs = self.index.search(q_emb, len(self.chunks))
        vec_score_map = {int(i): float(s) for s, i in zip(vec_scores[0], vec_idxs[0]) if i != -1}

        # bm25 scores over the whole corpus
        bm25_scores = self.bm25.get_scores(self._tokenize(query))
        bm25_max = max(bm25_scores) if len(bm25_scores) and max(bm25_scores) > 0 else 1.0

        combined = []
        for idx in range(len(self.chunks)):
            v = vec_score_map.get(idx, 0.0)  # already in [-1, 1], mostly [0,1]
            b = bm25_scores[idx] / bm25_max  # normalize to [0,1]
            combined_score = alpha * v + (1 - alpha) * b
            combined.append((combined_score, idx))

        combined.sort(key=lambda x: x[0], reverse=True)
        top = combined[:k]
        return [RetrievedChunk(chunk=self.chunks[i], score=s) for s, i in top]


# --------------------------------------------------------------------------
# 5. LLM answer generation (Groq API)
# --------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a precise AI assistant that answers questions strictly from provided context."

ANSWER_PROMPT_TEMPLATE = """You are an AI assistant.
Answer only using the provided context.
If the answer is unavailable, say:
"I couldn't find this information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:"""


def format_context(retrieved: List[RetrievedChunk]) -> str:
    """Formats retrieved chunks into a single context string with source tags."""
    parts = []
    for i, r in enumerate(retrieved, start=1):
        parts.append(
            f"[Source {i}: {r.chunk.source}, page {r.chunk.page}]\n{r.chunk.text}"
        )
    return "\n\n".join(parts)


class GroqLLM:
    """Thin wrapper around the Groq chat-completions API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        if Groq is None:
            raise ImportError("groq is not installed. `pip install groq`.")
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "No Groq API key found. Set the GROQ_API_KEY environment variable "
                "or pass api_key explicitly."
            )
        self.client = Groq(api_key=api_key)
        self.model = model

    def answer(
        self,
        question: str,
        retrieved: List[RetrievedChunk],
        chat_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.2,
    ) -> str:
        context = format_context(retrieved)
        prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if chat_history:
            # include prior turns for conversational memory (bonus feature)
            messages.extend(chat_history)
        messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content.strip()


# --------------------------------------------------------------------------
# 6. Convenience: keyword highlighting (bonus feature)
# --------------------------------------------------------------------------

def highlight_keywords(text: str, query: str) -> str:
    """
    Wraps query keywords found in `text` with **markdown bold** so the
    Streamlit UI can visually highlight matched terms in the retrieved
    context shown to the user.
    """
    keywords = sorted(set(re.findall(r"\w+", query.lower())), key=len, reverse=True)
    keywords = [w for w in keywords if len(w) > 2]  # skip tiny stopword-like tokens
    if not keywords:
        return text

    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in keywords) + r")\b", re.IGNORECASE)
    return pattern.sub(lambda m: f"**{m.group(0)}**", text)
