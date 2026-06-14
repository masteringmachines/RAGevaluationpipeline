# RAG Retrieval Evaluation Pipeline

Offline evaluation harness for comparing retrievers (dense embeddings, BM25, or
hybrid) on a fixed corpus + query set, using **Recall@k**, **MRR@10**, and
**Hit@1**. No LLM API calls — everything runs locally (NFR1).

## Quickstart

```bash
pip install -e .
# or: pip install -r requirements.txt

python -m rag_eval.cli \
  --documents data/documents.jsonl \
  --queries data/queries.jsonl \
  --retrievers bm25,dense \
  --output-dir results
```

This writes `results/summary.json`, `results/{name}_per_query.csv`, and
`results/run_config.json`, and prints a comparison table to the console.

## Project layout

```
rag-eval-pipeline/
├── src/rag_eval/
│   ├── cli.py             # entry point (`rag-eval` console script)
│   ├── dataset.py          # load documents.jsonl / queries.jsonl
│   ├── metrics.py           # recall@k, MRR@10, hit@1, aggregation
│   ├── evaluator.py         # index build + evaluation loop
│   ├── report.py            # CSV / JSON / console table output
│   ├── cache.py             # embedding cache (model + corpus hash)
│   └── retrievers/
│       ├── base.py          # `Retriever` ABC — subclass this
│       ├── dense.py          # sentence-transformers + FAISS
│       └── sparse.py         # BM25 (rank_bm25)
├── data/                     # sample corpus + queries
├── tests/                    # pytest unit tests
└── docs/PRD.md               # original product spec
```

## Input formats

**`documents.jsonl`** — one JSON object per line:

```json
{"id": "doc1", "text": "..."}
```

**`queries.jsonl`** — one JSON object per line:

```json
{"query": "...", "relevant_doc_ids": ["doc1", "doc3"], "metadata": {}}
```

`--documents` also accepts a directory of `.txt` (filename stem = doc id) or
`.json` files (each containing `{"id": ..., "text": ...}`), per FR2.

## Adding a retriever (NFR3)

Subclass `Retriever` (`retrievers/base.py`) and implement `build_index` +
`search`, then register it in `retrievers/__init__.py`'s `REGISTRY`. See
`retrievers/sparse.py` for a minimal (~20 line) example. `cache.py` shows the
pattern for disk-caching an expensive index build, used by the dense retriever.

## Metrics

| Metric    | Definition                                                              |
|-----------|--------------------------------------------------------------------------|
| Recall@k  | Fraction of relevant docs found in the top-k results                    |
| MRR@10    | Mean of 1/rank of the first relevant doc (0 if none in top 10)          |
| Hit@1     | 1 if the top result is relevant, else 0                                 |

## Reproducibility (NFR4)

The dense retriever seeds NumPy (`--seed`, default 42) and caches embeddings
in `.cache/embeddings/`, keyed by model name + a hash of the document corpus —
delete the cache directory to force recomputation. `results/run_config.json`
records the exact CLI arguments used for each run.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
