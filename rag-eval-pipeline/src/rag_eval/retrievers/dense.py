"""Dense retriever: sentence-transformers embeddings + FAISS cosine search (FR4)."""
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..cache import fingerprint, load_embeddings, save_embeddings
from .base import Retriever


class DenseRetriever(Retriever):
    """all-MiniLM-L6-v2 (or any sentence-transformers model) + FAISS IndexFlatIP.

    Embeddings are L2-normalized so inner product == cosine similarity.
    Embeddings are cached to disk keyed by `cache.fingerprint` (FR7); delete
    the cache directory to force recomputation after corpus changes.
    """

    name = "dense"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = ".cache/embeddings",
        seed: int = 42,
    ) -> None:
        np.random.seed(seed)  # NFR4: reproducible index construction
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache_dir = Path(cache_dir)
        self.index: Optional[faiss.Index] = None
        self.doc_ids: List[str] = []

    def build_index(self, documents: Dict[str, str]) -> None:
        self.doc_ids = list(documents.keys())
        key = fingerprint(self.model_name, documents)

        embeddings = load_embeddings(self.cache_dir, key)
        if embeddings is None:
            embeddings = self.model.encode(
                [documents[d] for d in self.doc_ids],
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype("float32")
            save_embeddings(self.cache_dir, key, embeddings)

        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 10) -> List[str]:
        q = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        _, idx = self.index.search(q, top_k)
        return [self.doc_ids[i] for i in idx[0] if i != -1]
