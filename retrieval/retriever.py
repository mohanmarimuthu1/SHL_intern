"""
Semantic retrieval over the SHL catalog FAISS index.
Loaded once at startup; all queries are thread-safe reads.
"""

import pickle
from pathlib import Path
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path(__file__).parent / "faiss.index"
METADATA_PATH = Path(__file__).parent / "metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

# Valid catalog URLs — built at load time for O(1) lookup
_VALID_URLS: set[str] = set()


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(METADATA_PATH, "rb") as f:
            self.catalog: list[dict] = pickle.load(f)

        global _VALID_URLS
        _VALID_URLS = {item["url"] for item in self.catalog}

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_types: list[str] | None = None,
        remote_only: bool = False,
    ) -> list[dict]:
        """
        Semantic search over the catalog.

        Args:
            query: Natural language description of the role/need.
            top_k: Max results to return.
            filter_types: If set, only return items with at least one matching test type.
            remote_only: If True, only return remote-testing items.

        Returns:
            List of catalog dicts, ranked by relevance.
        """
        vec = self.model.encode([query], show_progress_bar=False)
        vec = np.array(vec, dtype="float32")
        faiss.normalize_L2(vec)

        # Fetch extra candidates to allow for post-filtering
        fetch_k = min(top_k * 5, self.index.ntotal)
        scores, indices = self.index.search(vec, fetch_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            item = self.catalog[idx]

            if remote_only and not item.get("remote_testing"):
                continue

            if filter_types:
                item_types = set(item.get("test_types", []))
                if not item_types.intersection(filter_types):
                    continue

            results.append({**item, "_score": float(score)})

            if len(results) >= top_k:
                break

        return results

    def get_by_name(self, name: str) -> dict | None:
        """Exact or fuzzy name lookup for comparison queries."""
        name_lower = name.lower().strip()
        # Exact match first
        for item in self.catalog:
            if item["name"].lower() == name_lower:
                return item
        # Partial match
        for item in self.catalog:
            if name_lower in item["name"].lower():
                return item
        return None

    def is_valid_url(self, url: str) -> bool:
        return url in _VALID_URLS

    def get_all(self) -> list[dict]:
        return self.catalog


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Singleton — loaded once at startup."""
    return Retriever()
