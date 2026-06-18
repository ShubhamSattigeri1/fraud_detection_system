"""
embedder.py
────────────
Reads all knowledge base .txt files, chunks them by rule/pattern/case,
embeds each chunk using SentenceTransformers, and saves a FAISS index
to disk so the retriever can query it instantly.

Run this ONCE to build the index:
    python embedder.py

Requirements:
    pip install faiss-cpu sentence-transformers
"""

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
KB_FILES = [
    "KB/kb_rbi_guidelines.txt",
    "KB/kb_pmla_aml.txt",
    "KB/kb_upi_fraud_patterns.txt",
    "KB/kb_soc_notes.txt",
]

INDEX_PATH    = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"
EMBED_MODEL   = "all-MiniLM-L6-v2"


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_document(filepath: str) -> list:
    filename = os.path.basename(filepath)
    chunks   = []

    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]

    for i, block in enumerate(blocks):
        if block.startswith("DOCUMENT:") or block.startswith("SOURCE:"):
            continue

        chunk_id = f"{filename}__chunk_{i}"
        chunks.append({
            "text":     block,
            "source":   filename,
            "chunk_id": chunk_id,
        })

    print(f"  [{filename}] -> {len(chunks)} chunks")
    return chunks


# ── Embedding + Indexing ──────────────────────────────────────────────────────

def build_index():
    print("\n[FraudShield] Loading embedding model ...")
    model = SentenceTransformer(EMBED_MODEL)

    all_chunks = []
    print("\n[FraudShield] Chunking knowledge base files ...")
    for filepath in KB_FILES:
        if not os.path.exists(filepath):
            print(f"  WARNING: {filepath} not found — skipping")
            continue
        all_chunks.extend(chunk_document(filepath))

    print(f"\n[FraudShield] Total chunks to embed: {len(all_chunks)}")

    texts      = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n[FraudShield] FAISS index saved -> {INDEX_PATH}")
    print(f"[FraudShield] Metadata saved    -> {METADATA_PATH}")
    print(f"[FraudShield] Index size        -> {index.ntotal} vectors\n")


if __name__ == "__main__":
    build_index()
