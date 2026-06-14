.PHONY: install test run-demo

install:
	pip install -e ".[dev]"

test:
	pytest -q

run-demo:
	rag-eval \
		--documents data/documents.jsonl \
		--queries data/queries.jsonl \
		--retrievers bm25,dense \
		--output-dir results
