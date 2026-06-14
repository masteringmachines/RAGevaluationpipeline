from typing import Dict, List

from rag_eval.evaluator import evaluate_retriever
from rag_eval.retrievers.base import Retriever


class DummyRetriever(Retriever):
    """Returns documents in insertion order, ignoring the query text."""

    name = "dummy"

    def __init__(self):
        self.doc_ids: List[str] = []

    def build_index(self, documents: Dict[str, str]) -> None:
        self.doc_ids = list(documents)

    def search(self, query: str, top_k: int = 10) -> List[str]:
        return self.doc_ids[:top_k]


def test_evaluate_retriever():
    documents = {"d1": "a", "d2": "b", "d3": "c"}
    queries = [
        {"query": "q1", "relevant_doc_ids": ["d1"]},
        {"query": "q2", "relevant_doc_ids": ["d3"]},
    ]

    rows, summary = evaluate_retriever(DummyRetriever(), documents, queries, top_k=2, k_values=(1, 2))

    assert len(rows) == 2
    assert summary["retriever"] == "dummy"
    assert summary["num_queries"] == 2
    # q1's relevant doc (d1) is rank 1 -> mrr 1.0; q2's (d3) is outside top_k -> mrr 0.0
    assert summary["mean_mrr@10"] == 0.5
    assert summary["mean_recall@1"] == 0.5
    assert summary["mean_recall@2"] == 0.5
