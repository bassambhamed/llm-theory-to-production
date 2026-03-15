"""
FastAPI Backend — RAG Application API.
Exposes endpoints for ingestion, querying (classic + graph RAG),
and dashboard data (embeddings PCA, knowledge graph, evaluation).
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_engine import RAGEngine

# ─── Global engine instance ───────────────────────────────────────────────

engine = RAGEngine(data_dir="sample_data")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Auto-ingest documents on startup."""
    result = engine.ingest_all()
    print(f"[startup] Ingested: {result}")
    yield


app = FastAPI(
    title="RAG Pipeline API",
    description="Complete RAG system with Classic & Graph RAG — Lab 6",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request / Response Models ─────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str
    method: str = Field(default="hybrid", description="dense | sparse | hybrid")
    top_k: int = Field(default=5, ge=1, le=20)
    rerank_top_k: int = Field(default=3, ge=1, le=10)
    use_reranking: bool = True


class GraphQueryRequest(BaseModel):
    question: str
    top_k_vector: int = Field(default=3, ge=1, le=10)


# ─── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {"status": "ok", "message": "RAG Pipeline API — Lab 6"}


@app.post("/ingest")
def ingest():
    """Re-ingest all documents from sample_data/."""
    result = engine.ingest_all()
    return {"status": "ok", **result}


@app.post("/query")
def query(req: QueryRequest):
    """Classic RAG query (dense / sparse / hybrid + optional reranking)."""
    return engine.query(
        question=req.question,
        top_k=req.top_k,
        rerank_top_k=req.rerank_top_k,
        use_reranking=req.use_reranking,
        method=req.method,
    )


@app.post("/query/graph")
def query_graph(req: GraphQueryRequest):
    """Graph RAG query (knowledge graph + vector retrieval)."""
    return engine.query_graph_rag(
        question=req.question,
        top_k_vector=req.top_k_vector,
    )


@app.get("/embeddings")
def get_embeddings():
    """Get 2D PCA projection of chunk embeddings for visualization."""
    return engine.get_embeddings_pca()


@app.get("/embeddings/3d")
def get_embeddings_3d():
    """Get 3D PCA projection of chunk embeddings for visualization."""
    return engine.get_embeddings_pca_3d()


@app.get("/graph")
def get_graph():
    """Get knowledge graph data (nodes, edges, communities)."""
    return engine.get_graph_data()


@app.get("/stats")
def get_stats():
    """Get system statistics."""
    return engine.get_stats()


@app.get("/evaluate")
def evaluate():
    """Evaluate retrieval quality on a golden set."""
    return engine.evaluate_retrieval()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
