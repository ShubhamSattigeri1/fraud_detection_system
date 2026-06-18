"""
retriever.py
─────────────
Loads the FAISS index built by embedder.py and exposes one function:

    retrieve(query, top_k=3)

You pass in your plain English SHAP summary as the query.
It returns the top matching rules/cases from the knowledge base.

Usage — paste after your plain English output:
    from retriever import retrieve, print_retrieved_rules
    results = retrieve(plain["summary"])
    print_retrieved_rules(results)
"""

import os
import re
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH    = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"
EMBED_MODEL   = "all-MiniLM-L6-v2"

_model    = None
_index    = None
_metadata = None


def _load():
    global _model, _index, _metadata

    if _model is None:
        print("[FraudShield] Loading retriever model ...")
        _model = SentenceTransformer(EMBED_MODEL)

    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. Run embedder.py first."
            )
        _index = faiss.read_index(INDEX_PATH)

    if _metadata is None:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)


def _extract_rule_id(text: str) -> str:
    match = re.search(r"\b(RBI-\d+|PMLA-\d+|UPI-\d+|SOC-\d+)\b", text)
    return match.group(1) if match else "UNKNOWN"


def _source_label(source_filename: str) -> str:
    mapping = {
        "kb_rbi_guidelines.txt":     "RBI Master Directions",
        "kb_pmla_aml.txt":           "PMLA / AML Guidelines",
        "kb_upi_fraud_patterns.txt": "UPI Fraud Patterns",
        "kb_soc_notes.txt":          "SOC Investigation Notes",
    }
    return mapping.get(source_filename, source_filename)


def retrieve(query: str, top_k: int = 3) -> list:
    """
    Finds the most relevant rules from the knowledge base
    for a given plain English fraud explanation.

    Parameters
    ----------
    query  : str — your SHAP plain English summary sentence
    top_k  : int — how many results to return (default 3)

    Returns
    -------
    list of dicts: rule_id, source, text, score
    """
    _load()

    query_vec = _model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype="float32")

    scores, indices = _index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = _metadata[idx]
        results.append({
            "rule_id": _extract_rule_id(chunk["text"]),
            "source":  _source_label(chunk["source"]),
            "text":    chunk["text"],
            "score":   round(float(score), 4),
        })

    return results


def print_retrieved_rules(results: list):
    print("\n" + "=" * 55)
    print("  MATCHED POLICIES FROM KNOWLEDGE BASE")
    print("=" * 55)
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] {r['rule_id']} - {r['source']}  (score: {r['score']})")
        preview = r["text"][:200].replace("\n", " ")
        print(f"      {preview}...")
    print("\n" + "=" * 55 + "\n")
