# RAG Pipeline Application — Lab 6

Full-stack RAG (Retrieval-Augmented Generation) application built from the Lab 6 notebook. Includes Classic RAG and Graph RAG with professional dashboards.

## Architecture

```
┌─────────────────┐         ┌─────────────────────┐
│   Streamlit UI  │◄──────► │   FastAPI Backend    │
│   (port 8501)   │  HTTP   │    (port 8000)       │
│                 │         │                      │
│  • Chat (RAG)   │         │  • /query            │
│  • Embeddings   │         │  • /query/graph      │
│  • Graph Viz    │         │  • /embeddings       │
│  • Evaluation   │         │  • /graph            │
└─────────────────┘         │  • /evaluate         │
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │     RAG Engine        │
                            │                      │
                            │  • SentenceTransformer│
                            │  • ChromaDB (in-mem)  │
                            │  • BM25 index         │
                            │  • CrossEncoder       │
                            │  • NetworkX graph     │
                            │  • Ollama (LLM)       │
                            └──────────────────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| **Classic RAG** | Dense, sparse (BM25), and hybrid (RRF) retrieval with cross-encoder reranking |
| **Graph RAG** | Knowledge graph with entity extraction, community detection, local + global retrieval |
| **Embeddings Dashboard** | Interactive 2D/3D PCA visualization of chunk embeddings with Plotly |
| **Knowledge Graph Dashboard** | Interactive graph visualization with entity types, communities, and statistics |
| **Evaluation** | Retrieval metrics (Hit Rate, MRR) on a golden set comparing all methods |

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running with `glm-4.7-flash` model:
  ```bash
  # Install Ollama: https://ollama.com/download
  ollama pull glm-4.7-flash
  ollama serve  # keep running in background
  ```

## Setup

```bash
# From this directory (part-04-rag/rag-app/)
pip install -r requirements.txt

# Download ML models (run ONCE with internet access, then works offline)
python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
  SentenceTransformer('all-MiniLM-L6-v2'); \
  CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

## Running

Open **two terminals** from this directory:

**Terminal 1 — API server:**
```bash
python api.py
# or: uvicorn api:app --reload --port 8000
```

**Terminal 2 — Dashboard:**
```bash
streamlit run dashboard.py
```

The API starts on `http://localhost:8000` (with auto-ingestion on startup).
The dashboard opens at `http://localhost:8501`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/ingest` | Re-ingest documents from `sample_data/` |
| `POST` | `/query` | Classic RAG query (dense/sparse/hybrid + reranking) |
| `POST` | `/query/graph` | Graph RAG query (knowledge graph + vector) |
| `GET` | `/embeddings` | 2D PCA projection of chunk embeddings |
| `GET` | `/embeddings/3d` | 3D PCA projection of chunk embeddings |
| `GET` | `/graph` | Knowledge graph data (nodes, edges, communities) |
| `GET` | `/stats` | System statistics |
| `GET` | `/evaluate` | Retrieval evaluation on golden set |

### Example API call

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is self-attention?", "method": "hybrid"}'
```

## Sample Data

The `sample_data/` directory contains 4 Markdown documents about:
- `transformers.md` — Transformer architecture, self-attention, positional encoding
- `rag_systems.md` — RAG pipeline, chunking, hybrid retrieval, evaluation
- `vector_databases.md` — HNSW, vector similarity, ChromaDB, FAISS
- `finetuning.md` — LoRA, QLoRA, RLHF, DPO, alignment

## Dashboard Pages

1. **Chat** — Interactive RAG chat with Classic or Graph RAG mode, configurable retrieval settings, source citations
2. **Embeddings** — 2D and 3D PCA scatter plots colored by source document, variance explained, chunk distribution
3. **Knowledge Graph** — Interactive Plotly graph with entity types, edge labels, community detection, degree statistics
4. **Evaluation** — Side-by-side comparison of Dense/Sparse/Hybrid retrieval with Hit Rate and MRR metrics

## Models Used

| Component | Model |
|-----------|-------|
| Embeddings | `all-MiniLM-L6-v2` (384 dim) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `glm-4.7-flash` via Ollama (local) |
