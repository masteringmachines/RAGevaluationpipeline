import pytest

from rag_eval.metrics import (
    aggregate_metrics,
    compute_query_metrics,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = ["b", "d", "z"]
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, relevant, k=4) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, [], k=4) == 0.0


def test_hit_at_k():
    assert hit_at_k(["a", "b"], ["b"], k=1) == 0
    assert hit_at_k(["a", "b"], ["b"], k=2) == 1


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b", "c"], ["z"]) == 0.0


def test_compute_query_metrics_keys():
    m = compute_query_metrics(["a", "b"], ["a"], k_values=(1, 3))
    assert set(m) == {"recall@1", "recall@3", "mrr@10", "hit@1"}
    assert m["hit@1"] == 1
    assert m["recall@1"] == 1.0


def test_aggregate_metrics():
    rows = [{"recall@1": 1.0, "hit@1": 1}, {"recall@1": 0.0, "hit@1": 0}]
    agg = aggregate_metrics(rows)
    assert agg["mean_recall@1"] == pytest.approx(0.5)
    assert agg["mean_hit@1"] == pytest.approx(0.5)
    assert aggregate_metrics([]) == {}
