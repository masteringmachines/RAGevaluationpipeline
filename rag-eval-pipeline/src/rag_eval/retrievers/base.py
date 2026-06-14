"""Pluggable retriever interface (FR3). Subclass this for FR4 / NFR3."""
from abc import ABC, abstractmethod
from typing import Dict, List


class Retriever(ABC):
    """Minimal contract the evaluator depends on.

    Implement `build_index` once per evaluation run, then `search` for every
    query. `name` is used in report filenames and summary rows.
    """

    name: str = "retriever"

    @abstractmethod
    def build_index(self, documents: Dict[str, str]) -> None:
        """Build (or load a cached) search index over {doc_id: text}."""

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[str]:
        """Return up to `top_k` doc IDs, most relevant first."""
