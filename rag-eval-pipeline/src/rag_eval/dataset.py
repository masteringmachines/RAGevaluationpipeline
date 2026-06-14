"""Loaders for the corpus (FR2) and the query/relevance dataset (FR1)."""
import json
from pathlib import Path
from typing import Dict, List


def load_documents(path: str) -> Dict[str, str]:
    """Load {doc_id: text}.

    Accepts either a `documents.jsonl` file (one ``{"id": ..., "text": ...}``
    object per line) or a directory containing `.txt` files (filename stem
    used as the id) and/or `.json` files (each ``{"id": ..., "text": ...}``).
    """
    p = Path(path)
    docs: Dict[str, str] = {}

    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix == ".txt":
                docs[f.stem] = f.read_text(encoding="utf-8")
            elif f.suffix == ".json":
                obj = json.loads(f.read_text(encoding="utf-8"))
                docs[obj.get("id", f.stem)] = obj["text"]
        return docs

    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                obj = json.loads(line)
                docs[obj["id"]] = obj["text"]
    return docs


def load_queries(path: str) -> List[dict]:
    """Load a list of ``{"query": ..., "relevant_doc_ids": [...], "metadata": {...}}``."""
    queries: List[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries
