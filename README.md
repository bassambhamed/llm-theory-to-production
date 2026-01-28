# Transformers & Large Language Models - From Theory to Production

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📚 About This Course

A comprehensive, hands-on training program covering the evolution of Natural Language Processing from classical methods to modern Large Language Models, Retrieval-Augmented Generation (RAG), fine-tuning techniques, Model Context Protocol (MCP), and autonomous AI agents.

**Author:** Bassem Ben Hamed
**Affiliation:** Professor of Applied Mathematics at Sfax University & Tech Lead at Digital Innovation Partner
**Repository:** [https://github.com/bassambhamed/llm-theory-to-production](https://github.com/bassambhamed/llm-theory-to-production)

**Target Audience:** Academic Researchers, Data Scientists, Software Engineers, AI/ML Practitioners, and Technical Managers.

**Format:** Mixed theoretical lectures with practical hands-on labs.

## 🎯 Learning Objectives

By completing this course, participants will:

1. ✅ Understand the evolution from classical NLP to modern LLMs
2. ✅ Master Transformer architectures and their variants
3. ✅ Build production-ready RAG systems
4. ✅ Apply fine-tuning techniques (LoRA, QLoRA, DPO)
5. ✅ Implement standardized tool integration with MCP
6. ✅ Design and deploy autonomous AI agent systems
7. ✅ Evaluate and optimize LLM performance

## 📋 Course Structure

### Part 1: NLP Fundamentals
- Classical NLP techniques (N-grams, TF-IDF, Word2Vec)
- Text representation and evaluation metrics
- **Lab:** Building classical NLP models

### Part 2: From RNNs to Transformers
- RNN, LSTM, GRU architectures
- Transformer architecture deep dive
- Attention mechanisms
- **Labs:** RNN vs Transformer comparison

### Part 3: Large Language Models
- GPT family evolution and open-source ecosystem
- Pre-training and scaling laws
- Autoregressive generation
- **Lab:** Interacting with foundation models

### Part 4: Retrieval-Augmented Generation (RAG)
- RAG architecture and embeddings
- Vector databases and semantic search
- Advanced techniques (chunking, reranking, hybrid search)
- **Lab:** Production RAG systems

### Part 5: Fine-Tuning & Adaptation
- Supervised Fine-Tuning (SFT)
- Parameter-Efficient Fine-Tuning (LoRA, QLoRA)
- Alignment techniques (RLHF, DPO)
- Model evaluation and deployment
- **Labs:** Fine-tuning with PEFT methods

### Part 6: Model Context Protocol (MCP)
- Standardized LLM-tool integration
- Building MCP servers and clients
- **Labs:** Custom MCP implementations

### Part 7: Agentic AI
- Agent foundations and reasoning
- LangChain, LangGraph orchestration
- Multi-agent systems
- **Labs:** Building production agents

## 🛠️ Tech Stack

### Core Libraries
- **Deep Learning:** PyTorch, TensorFlow
- **Transformers:** Hugging Face Transformers, PEFT, TRL
- **LLM Frameworks:** LangChain, LlamaIndex, LangGraph
- **Agents:** CrewAI, AutoGen
- **Vector Databases:** Chroma, Weaviate, Pinecone

### Deployment & Optimization
- **Inference:** vLLM, Ollama, TGI
- **Quantization:** GPTQ, AWQ, GGUF
- **Evaluation:** lm-evaluation-harness, Ragas
- **Observability:** LangSmith, Weights & Biases

## 📂 Repository Structure

```
├── part-01-nlp-fundamentals/
│   ├── theory/
│   ├── labs/
│   └── notebooks/
├── part-02-rnn-to-transformers/
│   ├── theory/
│   ├── labs/
│   └── notebooks/
├── part-03-llms/
│   ├── theory/
│   ├── labs/
│   └── notebooks/
├── part-04-rag/
│   ├── theory/
│   ├── labs/
│   └── notebooks/
├── part-05-finetuning/
│   ├── theory/
│   ├── labs/
│   └── notebooks/
├── part-06-mcp/
│   ├── theory/
│   ├── labs/
│   └── notebooks/
├── part-07-agents/
│   ├── theory/
│   ├── labs/
│   └── notebooks/
├── datasets/
├── resources/
├── docs/
└── src/
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Conda (Anaconda)
- Git

### Installation with Conda

```bash
# Clone the repository
git clone https://github.com/bassambhamed/llm-theory-to-production.git
cd llm-theory-to-production

# Create conda environment from file
conda env create -f environment.yml

# Activate environment
conda activate llm
```

### Environment Setup

Create a `.env` file:

```bash
# Copy template
cp .env.example .env

# Edit with your API keys
# Minimum required:
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

### Verify Installation

```bash
python verify_setup.py
```

### Start Learning

```bash
# Launch Jupyter Lab
jupyter lab

# Navigate to part-01-nlp-fundamentals/notebooks/
```

## 📖 Documentation

- [Setup Guide](docs/setup.md) - Detailed installation instructions
- [Agents](docs/agents.md) - AI agents for course assistance
- [Skills](docs/skills.md) - Learning outcomes and competencies
- [Commands](docs/commands.md) - Useful CLI commands
- [Rules](docs/rules.md) - Best practices and conventions
- [Contributing](docs/contributing.md) - Contribution guidelines

## 🎓 Prerequisites

**Required:**
- Intermediate Python programming
- Basic Machine Learning concepts
- Familiarity with Jupyter Notebooks

**Recommended:**
- Linear algebra fundamentals
- Prior experience with PyTorch/TensorFlow
- Basic understanding of APIs and REST

## 📊 Datasets

The course uses various public datasets:
- **Classical NLP:** IMDB, AG News, CoNLL 2003
- **Instruction Tuning:** Alpaca, Dolly, OpenAssistant
- **Evaluation:** MMLU, HumanEval, MT-Bench
- **RAG:** ArXiv papers, Wikipedia dumps

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](docs/contributing.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🔗 Resources

### Academic Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Brown et al., 2020)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [RAG](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)

### Online Courses
- [Stanford CS224N](http://web.stanford.edu/class/cs224n/)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- [deeplearning.ai LLM Specialization](https://www.deeplearning.ai/)

### Community
- [Hugging Face Discord](https://huggingface.co/join/discord)
- [LangChain Discord](https://discord.gg/langchain)

## 📧 Contact

For questions or feedback:
- **Author:** Bassem Ben Hamed
- **Email:** [bassem.benhamed@example.com]
- **GitHub Issues:** [Open an issue](../../issues)
- **Discussion Forum:** [GitHub Discussions](../../discussions)

## 🙏 Acknowledgments

Special thanks to:
- Hugging Face for Transformers library
- OpenAI, Anthropic, Meta for foundational research
- LangChain team for agent frameworks
- All contributors and students

---

**⭐ Star this repository if you find it helpful!**

**📢 Share with others who might benefit from this course!**
