"""Retriever Registry (FR3/FR4): name -> Retriever subclass.

Each retriever's third-party dependencies are imported defensively, so e.g.
evaluating only BM25 doesn't require sentence-transformers/FAISS to be
installed (and vice versa).
"""
from .base import Retriever

REGISTRY = {}
__all__ = ["Retriever", "REGISTRY"]

try:
    from .sparse import BM25Retriever

    REGISTRY["bm25"] = BM25Retriever
    __all__.append("BM25Retriever")
except ImportError:
    pass

try:
    from .dense import DenseRetriever

    REGISTRY["dense"] = DenseRetriever
    __all__.append("DenseRetriever")
except ImportError:
    pass
