"""
RAG Engine — Core logic extracted from Lab 6 notebook.
Handles: ingestion, chunking, embeddings, vector/sparse/hybrid retrieval,
reranking, prompt building, Graph RAG (entities, knowledge graph, communities).
"""

import os
import re
import sys
import json
import hashlib
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi
import ollama

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
warnings.filterwarnings("ignore")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _load_model(loader, model_name: str):
    """Load a HuggingFace model, trying offline cache first then online."""
    # Try offline first (use local cache, no network call)
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        model = loader(model_name)
        return model
    except Exception:
        pass
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)

    # Fallback: download from HuggingFace Hub
    try:
        model = loader(model_name)
        return model
    except Exception as e:
        print(
            f"\n[ERROR] Cannot load model '{model_name}'.\n"
            f"  → {e}\n\n"
            f"  The model is not in the local cache and HuggingFace Hub is unreachable.\n"
            f"  Fix: run this ONCE with internet access to download the models:\n\n"
            f"    python -c \"from sentence_transformers import SentenceTransformer, CrossEncoder; "
            f"SentenceTransformer('{EMBED_MODEL_NAME}'); "
            f"CrossEncoder('{RERANKER_MODEL_NAME}')\"\n\n"
            f"  After that, the app works fully offline.\n",
            file=sys.stderr,
        )
        sys.exit(1)

# ─── Data Classes ──────────────────────────────────────────────────────────


@dataclass
class Document:
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.text[:200].encode()).hexdigest()[:8]


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.text[:100].encode()).hexdigest()[:8]


@dataclass
class Entity:
    name: str
    entity_type: str
    description: str = ""


@dataclass
class Relationship:
    source: str
    target: str
    relation: str
    description: str = ""


# ─── RAG Engine ────────────────────────────────────────────────────────────


class RAGEngine:
    """Complete RAG pipeline: ingestion → chunking → indexing → retrieval → generation."""

    LLM_MODEL = "glm-4.7-flash:latest"

    def __init__(self, data_dir: str = "sample_data"):
        self.data_dir = Path(data_dir)
        self.documents: List[Document] = []
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_texts: List[str] = []

        # Models (try local cache first, then download)
        self.embed_model = _load_model(SentenceTransformer, EMBED_MODEL_NAME)
        self.reranker = _load_model(CrossEncoder, RERANKER_MODEL_NAME)

        # Indices
        self.chroma_client = chromadb.Client()
        self.collection = None
        self.bm25_index = None

        # Graph RAG
        self.kg: Optional[nx.DiGraph] = None
        self.communities: List[Dict] = []
        self.all_entities: List[Entity] = []
        self.all_relationships: List[Relationship] = []

        # State
        self._ingested = False

    # ── Ingestion ──────────────────────────────────────────────────────

    def ingest_markdown(self, path: str) -> List[Document]:
        text = Path(path).read_text(encoding="utf-8")
        sections = []
        current_heading = "Introduction"
        current_lines = []

        for line in text.split("\n"):
            if line.startswith("## "):
                if current_lines:
                    sections.append((current_heading, "\n".join(current_lines).strip()))
                current_heading = line.lstrip("# ").strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_heading, "\n".join(current_lines).strip()))

        docs = []
        for heading, body in sections:
            if body.strip():
                docs.append(Document(
                    text=body,
                    metadata={
                        "source": Path(path).name,
                        "format": "markdown",
                        "section": heading,
                    },
                ))
        return docs

    def ingest_all(self) -> Dict:
        self.documents = []
        for md_file in sorted(self.data_dir.glob("*.md")):
            docs = self.ingest_markdown(str(md_file))
            self.documents.extend(docs)

        self._chunk_documents()
        self._embed_chunks()
        self._build_indices()
        self._build_graph()
        self._ingested = True

        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "graph_nodes": self.kg.number_of_nodes() if self.kg else 0,
            "graph_edges": self.kg.number_of_edges() if self.kg else 0,
            "communities": len(self.communities),
        }

    # ── Chunking ───────────────────────────────────────────────────────

    def _chunk_recursive(self, text: str, chunk_size: int = 256,
                         chunk_overlap: int = 50) -> List[str]:
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks_out = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks_out.append(chunk_text)
            start += chunk_size - chunk_overlap
        return chunks_out

    def _chunk_documents(self):
        self.chunks = []
        for doc in self.documents:
            texts = self._chunk_recursive(doc.text, chunk_size=256, chunk_overlap=50)
            for i, text in enumerate(texts):
                self.chunks.append(Chunk(
                    text=text,
                    metadata={**doc.metadata, "chunk_index": str(i)},
                ))
        self.chunk_texts = [c.text for c in self.chunks]

    # ── Embeddings ─────────────────────────────────────────────────────

    def _embed_chunks(self):
        self.embeddings = self.embed_model.encode(
            self.chunk_texts, show_progress_bar=False, normalize_embeddings=True
        )

    # ── Indices ────────────────────────────────────────────────────────

    def _build_indices(self):
        # ChromaDB
        try:
            self.chroma_client.delete_collection("rag_app")
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name="rag_app",
            metadata={"hnsw:space": "cosine"},
        )
        self.collection.add(
            ids=[c.chunk_id for c in self.chunks],
            embeddings=self.embeddings.tolist(),
            documents=self.chunk_texts,
            metadatas=[c.metadata for c in self.chunks],
        )

        # BM25
        tokenized = [text.lower().split() for text in self.chunk_texts]
        self.bm25_index = BM25Okapi(tokenized)

    # ── Retrieval ──────────────────────────────────────────────────────

    def retrieve_dense(self, query: str, top_k: int = 5) -> List[Dict]:
        query_emb = self.embed_model.encode([query], normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"text": doc, "metadata": meta, "score": round(1 - dist, 4), "method": "dense"}
            for doc, meta, dist in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
        ]

    def retrieve_sparse(self, query: str, top_k: int = 5) -> List[Dict]:
        query_tokens = query.lower().split()
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"text": self.chunks[i].text, "metadata": self.chunks[i].metadata,
             "score": round(float(scores[i]), 4), "method": "sparse"}
            for i in top_indices if scores[i] > 0
        ]

    def retrieve_hybrid(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Dict]:
        dense_results = self.retrieve_dense(query, top_k=top_k * 2)
        sparse_results = self.retrieve_sparse(query, top_k=top_k * 2)

        rrf_scores = {}
        doc_map = {}

        for rank, r in enumerate(dense_results):
            key = r["text"][:100]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rrf_k + rank + 1)
            doc_map[key] = r

        for rank, r in enumerate(sparse_results):
            key = r["text"][:100]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rrf_k + rank + 1)
            doc_map[key] = r

        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [
            {**doc_map[k], "score": round(rrf_scores[k], 4), "method": "hybrid"}
            for k in sorted_keys
        ]

    def rerank(self, query: str, results: List[Dict], top_k: int = 3) -> List[Dict]:
        if not results:
            return []
        pairs = [(query, r["text"]) for r in results]
        scores = self.reranker.predict(pairs)
        for r, score in zip(results, scores):
            r["rerank_score"] = round(float(score), 4)
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    # ── Prompt & Generation ────────────────────────────────────────────

    def build_rag_prompt(self, query: str, retrieved_chunks: List[Dict]) -> str:
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk["metadata"].get("source", "unknown")
            section = chunk["metadata"].get("section", "")
            chunk_id = f"{source}:{section}" if section else source
            context_blocks.append(f"[Source {i+1}: {chunk_id}]\n{chunk['text']}")

        context_str = "\n\n---\n\n".join(context_blocks)

        return f"""You are an expert assistant. Answer the user's question using ONLY the provided context.

Rules:
- Answer ONLY from the provided context below.
- If the evidence is insufficient, say exactly: "Insufficient evidence in the provided context."
- Each factual claim must cite at least one source using [Source N] format.
- Be concise and direct.

Context:
{context_str}

Question: {query}

Answer:"""

    def generate_answer(self, prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 500},
            )
            return response.message.content
        except Exception as e:
            return (
                f"[LLM call failed: {e}]\n\n"
                "Make sure Ollama is running (`ollama serve`) "
                f"and {self.LLM_MODEL} is installed (`ollama pull glm-4.7-flash`)."
            )

    # ── Full RAG Pipeline ──────────────────────────────────────────────

    def query(self, question: str, top_k: int = 5, rerank_top_k: int = 3,
              use_reranking: bool = True, method: str = "hybrid") -> Dict:
        if not self._ingested:
            return {"error": "No documents ingested. Call /ingest first."}

        if method == "dense":
            retrieved = self.retrieve_dense(question, top_k=top_k)
        elif method == "sparse":
            retrieved = self.retrieve_sparse(question, top_k=top_k)
        else:
            retrieved = self.retrieve_hybrid(question, top_k=top_k)

        if use_reranking and retrieved:
            final_chunks = self.rerank(question, retrieved, top_k=rerank_top_k)
        else:
            final_chunks = retrieved[:rerank_top_k]

        prompt = self.build_rag_prompt(question, final_chunks)
        answer = self.generate_answer(prompt)

        return {
            "query": question,
            "answer": answer,
            "sources": [
                {
                    "source": c["metadata"].get("source", ""),
                    "section": c["metadata"].get("section", ""),
                    "score": c.get("rerank_score", c.get("score", 0)),
                    "text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                }
                for c in final_chunks
            ],
            "method": method,
            "prompt_length": len(prompt.split()),
        }

    # ── Graph RAG ──────────────────────────────────────────────────────

    def _extract_entities_rules(self, text: str) -> Tuple[List[Entity], List[Relationship]]:
        known_entities = {
            "Transformer": "TECHNOLOGY", "BERT": "TECHNOLOGY", "GPT": "TECHNOLOGY",
            "T5": "TECHNOLOGY", "BART": "TECHNOLOGY", "RNN": "TECHNOLOGY",
            "LSTM": "TECHNOLOGY", "LoRA": "METHOD", "QLoRA": "METHOD",
            "RLHF": "METHOD", "DPO": "METHOD", "PPO": "METHOD",
            "RAG": "CONCEPT", "self-attention": "CONCEPT", "cross-attention": "CONCEPT",
            "fine-tuning": "CONCEPT", "embedding": "CONCEPT", "chunking": "CONCEPT",
            "BM25": "METHOD", "HNSW": "METHOD", "RoPE": "METHOD", "ALiBi": "METHOD",
            "hybrid retrieval": "CONCEPT", "reranking": "CONCEPT",
            "ChromaDB": "TECHNOLOGY", "Chroma": "TECHNOLOGY", "FAISS": "TECHNOLOGY",
            "Pinecone": "TECHNOLOGY", "Weaviate": "TECHNOLOGY",
            "Qdrant": "TECHNOLOGY", "Milvus": "TECHNOLOGY",
            "cosine similarity": "METRIC", "Recall@k": "METRIC",
            "MRR": "METRIC", "nDCG": "METRIC", "Faithfulness": "METRIC",
            "Ragas": "FRAMEWORK", "LangChain": "FRAMEWORK",
        }

        known_relations = [
            ("RAG", "embedding", "uses"), ("RAG", "chunking", "uses"),
            ("RAG", "BM25", "uses"), ("RAG", "reranking", "uses"),
            ("Transformer", "self-attention", "uses"),
            ("BERT", "Transformer", "is_a"), ("GPT", "Transformer", "is_a"),
            ("T5", "Transformer", "is_a"),
            ("HNSW", "FAISS", "part_of"), ("LoRA", "fine-tuning", "is_a"),
            ("QLoRA", "LoRA", "improves"), ("DPO", "RLHF", "improves"),
            ("hybrid retrieval", "BM25", "uses"),
            ("hybrid retrieval", "embedding", "uses"),
            ("Ragas", "RAG", "evaluates"),
            ("RoPE", "Transformer", "part_of"),
            ("cross-attention", "Transformer", "part_of"),
        ]

        text_lower = text.lower()
        found_entities = []
        for name, etype in known_entities.items():
            if name.lower() in text_lower:
                found_entities.append(Entity(name, etype))

        found_names = {e.name.lower() for e in found_entities}
        found_rels = []
        for src, tgt, rel in known_relations:
            if src.lower() in found_names and tgt.lower() in found_names:
                found_rels.append(Relationship(src, tgt, rel))

        return found_entities, found_rels

    @staticmethod
    def _canonicalize_name(name: str) -> str:
        canonical = name.strip()
        if not canonical.isupper():
            canonical = canonical.title()
        return canonical

    def _build_graph(self):
        self.all_entities = []
        self.all_relationships = []

        for doc in self.documents:
            entities, rels = self._extract_entities_rules(doc.text)
            self.all_entities.extend(entities)
            self.all_relationships.extend(rels)

        # Build NetworkX graph
        G = nx.DiGraph()
        seen = set()
        for e in self.all_entities:
            name = self._canonicalize_name(e.name)
            if name.lower() not in seen:
                G.add_node(name, type=e.entity_type, description=e.description)
                seen.add(name.lower())

        for r in self.all_relationships:
            src = self._canonicalize_name(r.source)
            tgt = self._canonicalize_name(r.target)
            if src.lower() in seen and tgt.lower() in seen:
                G.add_edge(src, tgt, relation=r.relation, description=r.description)

        self.kg = G

        # Detect communities
        if G.number_of_nodes() > 0:
            G_undirected = G.to_undirected()
            comms = list(greedy_modularity_communities(G_undirected))
            self.communities = []
            for i, comm in enumerate(comms):
                summary = self._summarize_community(comm)
                self.communities.append({
                    "community_id": i,
                    "entities": sorted(comm),
                    "summary": summary,
                })

    def _summarize_community(self, community: set) -> str:
        entities_info = []
        for node in community:
            data = self.kg.nodes[node]
            entities_info.append(f"- {node} ({data.get('type', 'UNKNOWN')})")

        relations_info = []
        for u, v, data in self.kg.edges(data=True):
            if u in community and v in community:
                relations_info.append(f"- {u} --[{data['relation']}]--> {v}")

        summary = f"Community with {len(community)} entities:\n"
        summary += "Entities:\n" + "\n".join(entities_info) + "\n"
        if relations_info:
            summary += "Relationships:\n" + "\n".join(relations_info)
        return summary

    def find_anchor_entities(self, query: str, threshold: float = 0.4) -> List[Tuple[str, float]]:
        if not self.kg or self.kg.number_of_nodes() == 0:
            return []
        query_emb = self.embed_model.encode([query], normalize_embeddings=True)
        node_names = list(self.kg.nodes())
        node_embs = self.embed_model.encode(node_names, normalize_embeddings=True)
        similarities = np.dot(query_emb, node_embs.T)[0]
        anchors = [(node_names[i], float(similarities[i]))
                    for i in range(len(node_names)) if similarities[i] > threshold]
        anchors.sort(key=lambda x: -x[1])
        return anchors

    def retrieve_local_graph(self, query: str, hops: int = 2,
                              max_entities: int = 5) -> Dict:
        anchors = self.find_anchor_entities(query)
        if not anchors:
            return {"entities": [], "triples": [], "context": "No relevant entities found."}

        relevant_nodes = set()
        for anchor_name, _ in anchors[:max_entities]:
            if anchor_name in self.kg:
                relevant_nodes.add(anchor_name)
                for _ in range(hops):
                    neighbors = set()
                    for n in relevant_nodes:
                        neighbors.update(self.kg.successors(n))
                        neighbors.update(self.kg.predecessors(n))
                    relevant_nodes.update(neighbors)

        triples = []
        for u, v, data in self.kg.edges(data=True):
            if u in relevant_nodes and v in relevant_nodes:
                triples.append({"source": u, "relation": data["relation"], "target": v})

        context_lines = [f"Entities: {', '.join(sorted(relevant_nodes))}"]
        context_lines.append("\nRelationships:")
        for t in triples:
            context_lines.append(f"  - {t['source']} --[{t['relation']}]--> {t['target']}")

        return {
            "entities": sorted(relevant_nodes),
            "triples": triples,
            "anchors": [(a, round(s, 3)) for a, s in anchors[:max_entities]],
            "context": "\n".join(context_lines),
        }

    def retrieve_global_graph(self, query: str, top_k: int = 2) -> Dict:
        if not self.communities:
            return {"communities": [], "context": "No communities detected."}

        query_emb = self.embed_model.encode([query], normalize_embeddings=True)
        summaries = [c["summary"] for c in self.communities]
        summary_embs = self.embed_model.encode(summaries, normalize_embeddings=True)

        similarities = np.dot(query_emb, summary_embs.T)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        relevant = []
        for idx in top_indices:
            relevant.append({
                "community_id": self.communities[idx]["community_id"],
                "similarity": round(float(similarities[idx]), 3),
                "summary": self.communities[idx]["summary"],
                "entities": self.communities[idx]["entities"],
            })

        context = "\n\n".join([
            f"[Community {c['community_id']+1}]\n{c['summary']}"
            for c in relevant
        ])

        return {"communities": relevant, "context": context}

    def query_graph_rag(self, question: str, top_k_vector: int = 3) -> Dict:
        if not self._ingested:
            return {"error": "No documents ingested. Call /ingest first."}

        local_result = self.retrieve_local_graph(question)
        global_result = self.retrieve_global_graph(question, top_k=2)

        vector_results = self.retrieve_hybrid(question, top_k=top_k_vector)
        vector_reranked = self.rerank(question, vector_results, top_k=top_k_vector)

        graph_context = (
            "=== Knowledge Graph Context ===\n"
            f"{local_result['context']}\n\n"
            "=== Community Summaries ===\n"
            f"{global_result['context']}"
        )

        vector_context = "\n\n---\n\n".join([
            f"[Chunk from {r['metadata'].get('source', '?')}]\n{r['text']}"
            for r in vector_reranked
        ])

        prompt = f"""You are an expert assistant with access to both a knowledge graph and document chunks.

Rules:
- Answer ONLY from the provided context (graph + documents).
- If evidence is insufficient, say: "Insufficient evidence in the provided context."
- Cite sources when possible.
- Leverage the knowledge graph for relationships between entities.
- Use document chunks for detailed explanations.

{graph_context}

=== Document Chunks ===
{vector_context}

Question: {question}

Answer:"""

        answer = self.generate_answer(prompt)

        return {
            "query": question,
            "answer": answer,
            "graph_entities": local_result["entities"],
            "graph_triples": local_result["triples"],
            "communities_used": len(global_result.get("communities", [])),
            "vector_chunks": len(vector_reranked),
            "method": "graph_rag",
        }

    # ── Data for dashboards ────────────────────────────────────────────

    def get_embeddings_pca(self, n_components: int = 2) -> Dict:
        if self.embeddings is None:
            return {"error": "No embeddings computed."}

        pca = PCA(n_components=n_components, random_state=42)
        coords = pca.fit_transform(self.embeddings)

        points = []
        for i, chunk in enumerate(self.chunks):
            points.append({
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "source": chunk.metadata.get("source", "unknown"),
                "section": chunk.metadata.get("section", ""),
                "text_preview": chunk.text[:120],
            })

        return {
            "points": points,
            "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "n_chunks": len(self.chunks),
            "embedding_dim": int(self.embeddings.shape[1]),
        }

    def get_embeddings_pca_3d(self) -> Dict:
        if self.embeddings is None:
            return {"error": "No embeddings computed."}

        pca = PCA(n_components=3, random_state=42)
        coords = pca.fit_transform(self.embeddings)

        points = []
        for i, chunk in enumerate(self.chunks):
            points.append({
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "z": round(float(coords[i, 2]), 4),
                "source": chunk.metadata.get("source", "unknown"),
                "section": chunk.metadata.get("section", ""),
                "text_preview": chunk.text[:120],
            })

        return {
            "points": points,
            "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        }

    def get_graph_data(self) -> Dict:
        if not self.kg:
            return {"error": "No knowledge graph built."}

        nodes = []
        for n, data in self.kg.nodes(data=True):
            nodes.append({
                "id": n,
                "type": data.get("type", "UNKNOWN"),
                "degree": self.kg.degree(n),
            })

        edges = []
        for u, v, data in self.kg.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "relation": data.get("relation", ""),
            })

        type_counts = {}
        for n in nodes:
            t = n["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "nodes": nodes,
            "edges": edges,
            "type_counts": type_counts,
            "communities": [
                {"id": c["community_id"], "entities": c["entities"], "size": len(c["entities"])}
                for c in self.communities
            ],
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }

    def get_stats(self) -> Dict:
        return {
            "ingested": self._ingested,
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings is not None else 0,
            "graph_nodes": self.kg.number_of_nodes() if self.kg else 0,
            "graph_edges": self.kg.number_of_edges() if self.kg else 0,
            "communities": len(self.communities),
            "llm_model": self.LLM_MODEL,
            "embed_model": EMBED_MODEL_NAME,
            "reranker_model": RERANKER_MODEL_NAME,
        }

    def evaluate_retrieval(self) -> Dict:
        golden_set = [
            {"query": "What is self-attention in transformers?",
             "expected_source": "transformers.md"},
            {"query": "What chunking strategies exist for RAG?",
             "expected_source": "rag_systems.md"},
            {"query": "What is LoRA fine-tuning?",
             "expected_source": "finetuning.md"},
            {"query": "How does HNSW indexing work?",
             "expected_source": "vector_databases.md"},
            {"query": "What are the benefits of hybrid retrieval?",
             "expected_source": "rag_systems.md"},
            {"query": "What is DPO in LLM alignment?",
             "expected_source": "finetuning.md"},
        ]

        methods = {
            "Dense": self.retrieve_dense,
            "Sparse (BM25)": self.retrieve_sparse,
            "Hybrid (RRF)": self.retrieve_hybrid,
        }

        results = {}
        for name, fn in methods.items():
            hit_rates = []
            mrrs = []
            for item in golden_set:
                retrieved = fn(item["query"], top_k=5)
                sources = [r["metadata"].get("source", "") for r in retrieved]
                hit = 1 if item["expected_source"] in sources else 0
                hit_rates.append(hit)
                try:
                    rank = sources.index(item["expected_source"]) + 1
                    mrrs.append(1.0 / rank)
                except ValueError:
                    mrrs.append(0.0)
            results[name] = {
                "hit_rate": round(float(np.mean(hit_rates)), 3),
                "mrr": round(float(np.mean(mrrs)), 3),
            }

        return {"methods": results, "golden_set_size": len(golden_set)}
