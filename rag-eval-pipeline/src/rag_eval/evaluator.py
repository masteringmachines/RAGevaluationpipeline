"""Evaluator Engine: build index once, score every query (data flow steps 2-5)."""
import time
from typing import Dict, List, Tuple

from .metrics import aggregate_metrics, compute_query_metrics
from .retrievers.base import Retriever


def evaluate_retriever(
    retriever: Retriever,
    documents: Dict[str, str],
    queries: List[dict],
    top_k: int = 10,
    k_values: Tuple[int, ...] = (1, 3, 5, 10),
) -> Tuple[List[Dict], Dict]:
    """Return (per_query_rows, summary) for one retriever.

    `per_query_rows` has one dict per query (query text, all metrics, and the
    retrieved doc ids) — ready for `report.write_csv`. `summary` aggregates
    those metrics plus index-build time and retriever identity (FR6/NFR4).
    """
    t0 = time.perf_counter()
    retriever.build_index(documents)
    build_time = time.perf_counter() - t0

    rows: List[Dict] = []
    per_query: List[Dict] = []
    for q in queries:
        retrieved = retriever.search(q["query"], top_k=top_k)
        m = compute_query_metrics(retrieved, q["relevant_doc_ids"], k_values)
        rows.append({"query": q["query"], **m, "retrieved": retrieved})
        per_query.append(m)

    summary = {
        "retriever": retriever.name,
        "num_queries": len(queries),
        "build_time_sec": build_time,
    }
    summary.update(aggregate_metrics(per_query))
    return rows, summary
