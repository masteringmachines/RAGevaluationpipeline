import json

from rag_eval.dataset import load_documents, load_queries


def test_load_documents_jsonl(tmp_path):
    f = tmp_path / "documents.jsonl"
    f.write_text(
        '{"id": "doc1", "text": "hello"}\n{"id": "doc2", "text": "world"}\n',
        encoding="utf-8",
    )
    assert load_documents(str(f)) == {"doc1": "hello", "doc2": "world"}


def test_load_documents_directory(tmp_path):
    (tmp_path / "doc1.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "doc2.json").write_text(json.dumps({"id": "doc2", "text": "world"}), encoding="utf-8")
    assert load_documents(str(tmp_path)) == {"doc1": "hello", "doc2": "world"}


def test_load_queries(tmp_path):
    f = tmp_path / "queries.jsonl"
    f.write_text('{"query": "q1", "relevant_doc_ids": ["doc1"]}\n', encoding="utf-8")
    assert load_queries(str(f)) == [{"query": "q1", "relevant_doc_ids": ["doc1"]}]
