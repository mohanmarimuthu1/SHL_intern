"""
Builds and persists a FAISS index over the SHL catalog.
Run once: python -m retrieval.embedder
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CATALOG_PATH = Path(__file__).parent.parent / "catalog" / "catalog.json"
INDEX_PATH = Path(__file__).parent / "faiss.index"
METADATA_PATH = Path(__file__).parent / "metadata.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"

TEST_TYPE_LABELS = {
    "A": "Ability and Aptitude cognitive reasoning",
    "B": "Biodata Situational Judgement behavioral",
    "C": "Competencies behavioral competency",
    "D": "Development 360 feedback development",
    "E": "Assessment Exercises",
    "K": "Knowledge Skills technical knowledge test",
    "P": "Personality Behavior personality questionnaire",
    "S": "Simulation work simulation",
}


def build_document(item: dict) -> str:
    """
    Build a rich text document for embedding.
    We concatenate name + test type descriptions to maximize semantic coverage.
    """
    name = item["name"]
    types = item.get("test_types", [])
    type_desc = " | ".join(TEST_TYPE_LABELS.get(t, t) for t in types)
    remote = "supports remote testing" if item.get("remote_testing") else ""
    adaptive = "adaptive IRT" if item.get("adaptive_irt") else ""
    extras = " ".join(filter(None, [remote, adaptive]))

    parts = [name, type_desc]
    if extras:
        parts.append(extras)
    if item.get("description"):
        parts.append(item["description"][:400])

    return " | ".join(parts)


def build_index():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    print(f"Loaded {len(catalog)} catalog items.")
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    documents = [build_document(item) for item in catalog]
    print("Encoding documents...")
    embeddings = model.encode(documents, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine after normalization
    index.add(embeddings)

    INDEX_PATH.parent.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(catalog, f)

    print(f"Index saved to {INDEX_PATH} ({index.ntotal} vectors, dim={dim})")


if __name__ == "__main__":
    build_index()
