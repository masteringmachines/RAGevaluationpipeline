"""On-disk embedding cache keyed by model name + corpus fingerprint (FR7)."""
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def fingerprint(model_name: str, documents: Dict[str, str]) -> str:
    """Stable hash over the model name and every (doc_id, text) pair.

    Changes whenever the corpus content/membership or the model changes,
    which invalidates the cache automatically.
    """
    h = hashlib.sha256(model_name.encode())
    for doc_id in sorted(documents):
        h.update(doc_id.encode())
        h.update(documents[doc_id].encode())
    return h.hexdigest()[:16]


def load_embeddings(cache_dir: Path, key: str) -> Optional[np.ndarray]:
    path = Path(cache_dir) / f"{key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_embeddings(cache_dir: Path, key: str, embeddings: np.ndarray) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / f"{key}.pkl", "wb") as f:
        pickle.dump(embeddings, f)
