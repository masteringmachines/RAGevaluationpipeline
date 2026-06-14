"""Lexical retriever using BM25 (FR4)."""
import re
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from .base import Retriever

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever(Retriever):
    """BM25 over whitespace/word tokens (rank_bm25). ~20 lines (NFR3)."""

    name = "bm25"

    def __init__(self) -> None:
        self.bm25: Optional[BM25Okapi] = None
        self.doc_ids: List[str] = []

    def build_index(self, documents: Dict[str, str]) -> None:
        self.doc_ids = list(documents.keys())
        tokenized = [_tokenize(documents[d]) for d in self.doc_ids]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> List[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.doc_ids[i] for i in ranked[:top_k]]
