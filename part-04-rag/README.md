# Part 4: Retrieval-Augmented Generation (RAG)

## 📚 Overview

Learn to ground LLMs with external knowledge sources to reduce hallucinations and enable domain-specific applications. Build complete RAG pipelines from document ingestion to production-ready "Chat with your Data" systems.

## 🎯 Learning Objectives

By the end of this part, you will:

1. ✅ Understand RAG architecture and why it complements pure LLMs
2. ✅ Master embedding models and vector similarity search
3. ✅ Implement document chunking strategies and hybrid search
4. ✅ Build reranking pipelines and query transformations
5. ✅ Explore advanced patterns: GraphRAG and Agentic RAG
6. ✅ Evaluate RAG systems with standardized metrics (Ragas)

## 📖 Theory Topics

### 4.1 Foundations of RAG

- **Limitations of Pure LLMs**
  - Knowledge cutoff dates and hallucinations
  - Inability to access private/proprietary data
  - Static knowledge base
- **RAG Architecture (Lewis et al. 2020)**
  - Retriever + Generator pipeline
  - Dense retrieval vs. sparse retrieval
  - End-to-end vs. modular approaches
- **Prompt Strategies for RAG**
  - Context injection and grounding techniques
  - RAG-specific prompt templates
  - Citation and source attribution in responses
- **Embedding Models**
  - OpenAI Ada (text-embedding-3-small/large)
  - Sentence-Transformers (all-MiniLM, all-mpnet)
  - Cohere Embed, Voyage AI, multilingual embeddings
- **Vector Similarity Metrics**
  - Cosine similarity, Euclidean distance, dot product
- **Vector Databases**
  - HNSW indexing
  - Chroma, Pinecone, Weaviate, Qdrant, Milvus, FAISS
  - Metadata filtering and hybrid search

### 4.2 Advanced RAG Techniques

- **Document Chunking Strategies**
  - Fixed-size (token-based, character-based)
  - Semantic chunking (sentence, paragraph)
  - Recursive splitting with overlap
  - Document-specific strategies (Markdown, PDF, code)
- **Hybrid Search**
  - Combining keyword search (BM25, TF-IDF) and vector search
  - Reciprocal Rank Fusion (RRF)
- **Reranking**
  - CrossEncoder models (Cohere Rerank, sentence-transformers)
  - LLM-based reranking and score fusion
- **Query Transformations**
  - Query expansion and reformulation
  - Hypothetical Document Embeddings (HyDE)
  - Multi-query retrieval
- **GraphRAG and Knowledge Graphs**
  - Entity extraction and relationship mapping
  - Graph-based retrieval and traversal
  - Combining vector search with graph queries
- **Agentic RAG**
  - Self-correcting RAG with reflection loops
  - Adaptive retrieval (deciding when to retrieve)
  - Multi-step retrieval and reasoning
  - Router-based RAG (choosing retrieval strategy dynamically)
- **RAG Evaluation**
  - Frameworks: Ragas, ARES
  - Metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy
  - End-to-end vs. component-level evaluation

## 🔬 Lab Exercises

### Lab 6: Building Production RAG Systems

**Objectives:**
- Ingest and process multi-format documents (PDFs, Markdown, Web pages)
- Implement different chunking strategies and compare performance
- Build a "Chat with your Data" application
- Implement hybrid search with reranking
- Add metadata filtering and source attribution
- Build a GraphRAG pipeline with entity extraction
- Implement agentic RAG with adaptive retrieval
- Evaluate RAG systems with Ragas (Faithfulness, Relevancy, Precision, Recall)
- Handle edge cases: retrieval failures, irrelevant context

**Tools:**
- LangChain, LlamaIndex, Vector Stores (Chroma, Weaviate), Ragas

**Duration:** 4-5 hours

## 📓 Notebooks

1. **lab6-rag-pipeline.ipynb** — Building Production RAG Systems

## 🎞️ Slides

- **part4-rag-fr.pdf**

## 🛠️ RAG Application

The `rag-app/` directory contains a standalone RAG application with:
- `rag_engine.py` — Core retrieval and generation engine
- `api.py` — FastAPI backend
- `dashboard.py` — Monitoring dashboard
- `requirements.txt` — Dependencies

## 🚀 Getting Started

```bash
# Activate environment
conda activate llm

# Start Jupyter
jupyter lab

# Open the notebook
# notebooks/lab6-rag-pipeline.ipynb

# Or run the RAG app
cd rag-app
pip install -r requirements.txt
python api.py
```

## 📋 Prerequisites

- Completion of Parts 1–3 (NLP Fundamentals, RNNs to Transformers, LLMs)
- Understanding of embeddings and transformer architectures
- API keys for OpenAI and/or Anthropic (for embedding and generation)
- Familiarity with prompt engineering (Part 3.2)

## 📚 Resources

### Papers
- Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Karpukhin et al. (2020) — Dense Passage Retrieval for Open-Domain Question Answering
- Izacard & Grave (2021) — Leveraging Passage Retrieval with Generative Models for Open Domain QA
- Gao et al. (2023) — Retrieval-Augmented Generation for Large Language Models: A Survey
- Edge et al. (2024) — From Local to Global: A Graph RAG Approach to Query-Focused Summarization

### Tutorials
- [LangChain Documentation](https://python.langchain.com/docs)
- [LlamaIndex Documentation](https://docs.llamaindex.ai)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [Ragas Evaluation Framework](https://docs.ragas.io)

### Datasets
- Sample documents provided in `notebooks/sample_data/`
- Wikipedia passages (for retrieval benchmarks)
- MS MARCO (passage retrieval evaluation)

## 🎓 Assessment

To complete this part:
- [ ] Complete Lab 6 — Building Production RAG Systems
- [ ] Understand the RAG architecture and retrieval pipeline
- [ ] Compare chunking strategies and their impact on retrieval quality
- [ ] Implement hybrid search with reranking
- [ ] Evaluate a RAG system using Ragas metrics
- [ ] Run the standalone RAG application

---

**Previous:** [Part 3: From Transformers to LLMs](../part-03-llms/) · **Next:** [Part 5: Fine-Tuning and Model Adaptation](../part-05-finetuning/)
