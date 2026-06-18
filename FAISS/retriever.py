import os, re, json, logging
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

INDEX_PATH    = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"
EMBED_MODEL   = "all-MiniLM-L6-v2"

_model    = None
_index    = None
_metadata = None


def _load():
    global _model, _index, _metadata
    if SentenceTransformer is None or faiss is None:
        raise RuntimeError("FAISS or SentenceTransformer not installed")

    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)

    if _index is None:
        path = os.environ.get("FAISS_INDEX_PATH", INDEX_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAISS index not found at {path}")
        _index = faiss.read_index(path)

    if _metadata is None:
        path = os.environ.get("FAISS_METADATA_PATH", METADATA_PATH)
        with open(path, "r", encoding="utf-8") as f:
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
    try:
        _load()
    except (RuntimeError, FileNotFoundError) as e:
        logging.warning(f"FAISS retrieve skipped: {e}")
        return []

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
