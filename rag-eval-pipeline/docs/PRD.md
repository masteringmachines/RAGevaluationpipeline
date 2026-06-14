# PRD #9 – RAG Retrieval Evaluation Pipeline

## 1. Objective
Build a reusable evaluation pipeline that measures the **retrieval quality** of a RAG system (e.g., vector database, BM25, or hybrid retriever) using standard information retrieval metrics: **Recall@k**, **Mean Reciprocal Rank (MRR)**, and **Hit Rate**.

## 2. Scope
**In scope**  
- Offline evaluation on a static corpus of documents and a set of query‑relevant document ID pairs.  
- Support for multiple retrievers (embedding‑based, lexical, hybrid) via a pluggable interface.  
- Generate human‑readable reports (CSV, JSON, console table).  
- Local execution – no external API calls required for evaluation (except possibly the retriever’s own index building).  

**Out of scope**  
- Online A/B testing of retrieval in production.  
- Automatic retriever hyperparameter tuning.  
- Real‑time streaming evaluation.

## 3. User Stories  
1. As an ML engineer, I want to compare two retrievers (e.g., sentence‑transformers vs. BM25) on my QA dataset so I can choose the best one.  
2. As a researcher, I want to see recall@1, recall@5, MRR@10 in a single table to quickly assess retrieval effectiveness.  
3. As a developer, I want to add a new retriever (e.g., Cohere embeddings) by implementing a simple `Retriever` interface without rewriting the evaluation loop.  

## 4. Functional Requirements  

| ID | Requirement |
|----|-------------|
| FR1 | Accept input datasets in JSONL format: each line has `query`, `relevant_doc_ids` (list), and optional `metadata`. |
| FR2 | Accept a directory of documents (plain text or JSON) with unique IDs. |
| FR3 | Provide an abstract base class `Retriever` with method `search(query: str, top_k: int) -> List[str]` (return doc IDs). |
| FR4 | Implement at least two concrete retrievers: **Dense** (sentence‑transformers + FAISS) and **Sparse** (BM25 via `rank_bm25`). |
| FR5 | Compute per‑query: recall@k (k=1,3,5,10), MRR@10, hit@1 (binary). |
| FR6 | Aggregate metrics across all queries: mean recall@k, mean MRR, hit rate. |
| FR7 | Cache document embeddings for the dense retriever to disk (pickle or Parquet) – avoid recomputing on every run. |
| FR8 | Output a CSV with one row per query (including all per‑query metrics) and a summary JSON with aggregated numbers. |

## 5. Non‑Functional Requirements  

| ID | Requirement |
|----|-------------|
| NFR1 | **Token efficiency** – Zero LLM API calls. All embeddings computed locally with a small model (`all‑MiniLM‑L6‑v2`). |
| NFR2 | **Performance** – Evaluate 1000 queries against 10k documents in < 5 minutes on a modern laptop. |
| NFR3 | **Modularity** – Adding a new retriever requires <20 lines of code (just subclass `Retriever`). |
| NFR4 | **Reproducibility** – Use fixed random seeds for any approximate index (e.g., FAISS IVF). Log configuration. |

## 6. System Design (High‑Level)

### Components
- **Dataset Loader** – reads queries + relevant IDs, and document store.  
- **Retriever Registry** – dictionary mapping retriever names to instances.  
- **Index Builder** (per retriever) – builds search index once before evaluation.  
- **Evaluator Engine** – loops over queries, calls retriever.search(), computes metrics.  
- **Cache Manager** – stores dense embeddings (`.pkl`) and FAISS index (`.faiss`).  
- **Report Generator** – writes CSV and JSON.

### Data Flow
1. Load `documents.jsonl` → in‑memory dict `{doc_id: text}`.  
2. For each retriever:  
   - If dense: load/cache embeddings, build FAISS index.  
   - If BM25: build tokenised corpus.  
3. Load `queries.jsonl`.  
4. For each query:  
   - Call `retriever.search(query, top_k=10)` → list of doc IDs.  
   - Compare against `relevant_doc_ids` → compute recall@k, MRR, hit.  
5. Aggregate & write reports.

### API Sketch (Python)
```python
class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[str]:
        pass

class DenseRetriever(Retriever):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.doc_ids = None

    def build_index(self, documents: Dict[str, str]):
        # compute embeddings, store in FAISS
        pass

    def search(self, query: str, top_k: int) -> List[str]:
        # embed query, FAISS search, return doc_ids
        pass

class BM25Retriever(Retriever):
    # similar interface