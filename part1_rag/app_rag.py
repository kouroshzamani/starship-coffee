from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

# =============================
# Environment
# =============================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "docs"
STORE_DIR = HERE / ".store"
STORE_DIR.mkdir(exist_ok=True)

JSON_STORE = STORE_DIR / "store.json"
SQLITE_STORE = STORE_DIR / "store.sqlite"

# =============================
# Chunking
# =============================
CHUNK_CHARS = 900
OVERLAP_CHARS = 120

# =============================
# Injection handling
# =============================
INJECTION_PATTERNS = [
    r"\bsecrets?\b",
    r"\breveal\b",
    r"\bdump\b",
    r"\bprint\b",
    r"\bexfiltrate\b",
    r"\bignore (all|previous) instructions\b",
]

REFUSAL_ONE_SENTENCE = (
    "I can’t help with requests to reveal secret files or contents; please ask a question about the public docs instead."
)


def is_injection(question: str) -> bool:
    q = question.lower()
    # extra guard for explicit paths
    if "secrets/" in q or "secrets\\" in q:
        return True
    return any(re.search(p, q, re.IGNORECASE) for p in INJECTION_PATTERNS)


# =============================
# Data model
# =============================
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    embedding: List[float]


# =============================
# TF-IDF globals
# =============================
_VECTORIZER: TfidfVectorizer | None = None


def ensure_vectorizer(chunks: List[Chunk]) -> None:
    """
    TF-IDF is corpus-dependent. If we loaded chunks from disk (json/sqlite),
    we must re-fit the vectorizer on the same chunk texts so queries live in
    the same feature space.
    """
    global _VECTORIZER
    if _VECTORIZER is not None:
        return

    texts = [c.text for c in chunks]
    _VECTORIZER = TfidfVectorizer(
        max_features=4096,
        stop_words="english",
    )
    _VECTORIZER.fit(texts)


def embed_corpus(texts: List[str]) -> np.ndarray:
    """
    Fit vectorizer on corpus and return embeddings for corpus texts.
    """
    global _VECTORIZER
    _VECTORIZER = TfidfVectorizer(
        max_features=4096,
        stop_words="english",
    )
    return _VECTORIZER.fit_transform(texts).toarray()


def embed_query(text: str) -> np.ndarray:
    assert _VECTORIZER is not None, "Vectorizer not initialized (call ensure_vectorizer or embed_corpus first)."
    return _VECTORIZER.transform([text]).toarray()[0]


# =============================
# Utils
# =============================
def read_markdown_docs() -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    if not DOCS_DIR.exists():
        return docs
    for p in sorted(DOCS_DIR.glob("*.md")):
        docs.append((p.name, p.read_text(encoding="utf-8", errors="ignore")))
    return docs


def chunk_text(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out: List[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + CHUNK_CHARS)
        piece = text[i:j].strip()
        if piece:
            out.append(piece)
        if j == len(text):
            break
        i = max(0, j - OVERLAP_CHARS)
    return out


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


# =============================
# Build chunks (fresh from docs)
# =============================
def build_chunks() -> List[Chunk]:
    docs = read_markdown_docs()
    texts: List[str] = []
    metas: List[Tuple[str, str]] = []

    for doc_id, content in docs:
        parts = chunk_text(content)
        for idx, part in enumerate(parts):
            texts.append(part)
            metas.append((doc_id, f"{doc_id}::c{idx}"))

    if not texts:
        return []

    vectors = embed_corpus(texts)  # FIT + EMBED corpus

    chunks: List[Chunk] = []
    for i, ((doc_id, chunk_id), text) in enumerate(zip(metas, texts)):
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=text,
                embedding=vectors[i].tolist(),
            )
        )
    return chunks


# =============================
# Storage
# =============================
def load_chunks(backend: str, rebuild: bool) -> List[Chunk]:
    backend = backend.lower()

    if backend == "json":
        if JSON_STORE.exists() and not rebuild:
            data = json.loads(JSON_STORE.read_text(encoding="utf-8"))
            chunks = [Chunk(**c) for c in data.get("chunks", [])]
            if chunks:
                ensure_vectorizer(chunks)  # <-- IMPORTANT
            return chunks

        chunks = build_chunks()
        JSON_STORE.write_text(
            json.dumps({"chunks": [c.__dict__ for c in chunks]}, indent=2),
            encoding="utf-8",
        )
        # build_chunks already fit vectorizer
        return chunks

    if backend == "sqlite":
        conn = sqlite3.connect(SQLITE_STORE)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                text TEXT,
                embedding TEXT
            )
            """
        )
        if rebuild:
            conn.execute("DELETE FROM chunks")
            conn.commit()

        cur = conn.execute("SELECT COUNT(*) FROM chunks")
        (count,) = cur.fetchone()

        if count and count > 0:
            rows = conn.execute(
                "SELECT chunk_id, doc_id, text, embedding FROM chunks"
            ).fetchall()
            conn.close()
            chunks = [Chunk(r[0], r[1], r[2], json.loads(r[3])) for r in rows]
            if chunks:
                ensure_vectorizer(chunks)  # <-- IMPORTANT
            return chunks

        chunks = build_chunks()
        conn.executemany(
            "INSERT INTO chunks VALUES (?,?,?,?)",
            [(c.chunk_id, c.doc_id, c.text, json.dumps(c.embedding)) for c in chunks],
        )
        conn.commit()
        conn.close()
        # build_chunks already fit vectorizer
        return chunks

    raise ValueError("Invalid backend (use json or sqlite)")


# =============================
# Retrieval
# =============================
def retrieve(question: str, chunks: List[Chunk], k: int) -> List[Dict[str, Any]]:
    # safety: if something loaded from disk and vectorizer got reset, recover
    ensure_vectorizer(chunks)

    q_vec = embed_query(question)
    scored = []
    for c in chunks:
        sim = cosine_sim(q_vec, np.array(c.embedding, dtype=np.float32))
        scored.append((sim, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "doc_id": c.doc_id,
            "chunk_id": c.chunk_id,
            "score": float(score),
            "text": c.text,
        }
        for score, c in scored[:k]
    ]


# =============================
# Answer with citations
# =============================
def answer_with_citations(question: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    citations: List[str] = []
    seen = set()
    for p in passages:
        if p["doc_id"] not in seen:
            seen.add(p["doc_id"])
            citations.append(p["doc_id"])

    context = "\n\n".join(
        f"[{i+1}] ({p['doc_id']}) {p['text']}"
        for i, p in enumerate(passages)
    )

    prompt = (
        "Answer in at most 100 words.\n"
        "Use ONLY the passages below.\n"
        "If the answer is not present, say you don't know.\n\n"
        f"Question:\n{question}\n\n"
        f"Passages:\n{context}\n\n"
        "Answer:"
    )

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.responses.create(model=OPENAI_MODEL, input=prompt)
    answer = resp.output_text.strip()

    out = {"answer": answer, "citations": citations}
    print(json.dumps(out, indent=2))
    return out


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Starship Coffee - Task 1", layout="wide")
st.title("Starship Coffee — Task 1: Simple RAG (Citations + Injection Handling)")

with st.sidebar:
    backend = st.selectbox("Storage backend", ["json", "sqlite"], index=0)
    k = st.number_input("k (top passages)", 1, 10, 5)
    rebuild = st.checkbox("Rebuild index", value=False)
    st.caption("Docs loaded from part1_rag/docs/")

question = st.text_input("Ask a question", "What are the store hours?")

if st.button("Ask"):
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY")
        st.stop()

    if is_injection(question):
        st.write(REFUSAL_ONE_SENTENCE)
        st.stop()

    with st.status("Loading documents..."):
        chunks = load_chunks(backend, rebuild)
        if not chunks:
            st.error("No docs found in part1_rag/docs/")
            st.stop()

    with st.status("Retrieving..."):
        top = retrieve(question, chunks, int(k))

    with st.status("Generating answer..."):
        out = answer_with_citations(question, top)

    st.subheader("Answer")
    st.write(out["answer"])

    st.subheader("Citations")
    st.dataframe([{"doc_id": c} for c in out["citations"]])

    with st.expander("Debug (top-k passages)"):
        for i, p in enumerate(top, 1):
            st.markdown(f"**{i}. {p['doc_id']}** (score={p['score']:.3f})")
            st.write(p["text"][:300] + ("…" if len(p["text"]) > 300 else ""))
