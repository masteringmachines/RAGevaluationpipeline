"""Command-line entry point: evaluate one or more retrievers and write reports.

Usage:
    python -m rag_eval.cli --documents data/documents.jsonl --queries data/queries.jsonl \
        --retrievers bm25,dense --output-dir results
"""
import argparse
import json
import sys
from pathlib import Path

from .dataset import load_documents, load_queries
from .evaluator import evaluate_retriever
from .report import print_table, write_csv, write_summary
from .retrievers import REGISTRY


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate RAG retriever(s) with Recall@k, MRR@10, and Hit@1."
    )
    p.add_argument("--documents", required=True, help="documents.jsonl path or a directory of documents")
    p.add_argument("--queries", required=True, help="queries.jsonl path")
    p.add_argument(
        "--retrievers",
        default="bm25,dense",
        help=f"Comma-separated retriever names: {', '.join(REGISTRY)}",
    )
    p.add_argument("--top-k", type=int, default=10, help="Documents to retrieve per query")
    p.add_argument("--k-values", default="1,3,5,10", help="Comma-separated k values for recall@k")
    p.add_argument("--output-dir", default="results", help="Directory for CSV/JSON reports")
    p.add_argument("--model-name", default="all-MiniLM-L6-v2", help="sentence-transformers model (dense)")
    p.add_argument("--cache-dir", default=".cache/embeddings", help="Embedding cache directory (dense)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    k_values = tuple(int(k) for k in args.k_values.split(","))
    retriever_names = [r.strip() for r in args.retrievers.split(",") if r.strip()]

    unknown = [r for r in retriever_names if r not in REGISTRY]
    if unknown:
        sys.exit(f"Unknown retriever(s): {unknown}. Available: {list(REGISTRY)}")

    documents = load_documents(args.documents)
    queries = load_queries(args.queries)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    summaries = []
    for name in retriever_names:
        cls = REGISTRY[name]
        kwargs = (
            {"model_name": args.model_name, "cache_dir": args.cache_dir, "seed": args.seed}
            if name == "dense"
            else {}
        )
        retriever = cls(**kwargs)

        rows, summary = evaluate_retriever(retriever, documents, queries, top_k=args.top_k, k_values=k_values)
        write_csv(rows, out_dir / f"{name}_per_query.csv")
        summaries.append(summary)
        print(
            f"[{name}] indexed {len(documents)} docs, evaluated {len(queries)} queries "
            f"(build time {summary['build_time_sec']:.2f}s)"
        )

    write_summary(summaries, out_dir / "summary.json")
    print()
    print_table(summaries)


if __name__ == "__main__":
    main()
