"""Information-retrieval metrics for FR5/FR6: per-query and aggregate."""
from typing import Dict, Iterable, List, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of `relevant` docs present in the top-k of `retrieved`."""
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / len(relevant)


def hit_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 1) -> int:
    """1 if any relevant doc appears in the top-k, else 0."""
    return int(bool(set(retrieved[:k]) & set(relevant)))


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str], k: int = 10) -> float:
    """1/rank of the first relevant doc within the top-k, else 0."""
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_query_metrics(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k_values: Iterable[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """All per-query metrics from FR5: recall@k for each k, MRR@10, hit@1."""
    metrics = {f"recall@{k}": recall_at_k(retrieved, relevant, k) for k in k_values}
    metrics["mrr@10"] = reciprocal_rank(retrieved, relevant, k=10)
    metrics["hit@1"] = hit_at_k(retrieved, relevant, k=1)
    return metrics


def aggregate_metrics(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each metric across all queries (FR6). Keys are prefixed with `mean_`."""
    if not per_query_metrics:
        return {}
    n = len(per_query_metrics)
    keys = per_query_metrics[0].keys()
    return {f"mean_{k}": sum(m[k] for m in per_query_metrics) / n for k in keys}
