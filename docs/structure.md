# Project Structure

Complete overview of the repository organization and file structure.

## 📁 Directory Tree

```
transformers-llms-course/
│
├── docs/                           # Documentation
│   ├── agents.md                   # AI agents documentation
│   ├── skills.md                   # Learning outcomes
│   ├── commands.md                 # CLI commands reference
│   ├── rules.md                    # Best practices
│   ├── setup.md                    # Setup instructions
│   ├── contributing.md             # Contribution guidelines
│   └── structure.md                # This file
│
├── part-01-nlp-fundamentals/       # Part 1: NLP Fundamentals
│   ├── theory/                     # Theoretical content
│   │   ├── 01-introduction.md
│   │   ├── 02-word-embeddings.md
│   │   └── 03-evaluation-metrics.md
│   ├── labs/                       # Lab exercises
│   │   ├── lab01-classical-nlp/
│   │   │   ├── README.md
│   │   │   ├── solution.py
│   │   │   └── tests.py
│   │   └── lab01-exercises.md
│   ├── notebooks/                  # Jupyter notebooks
│   │   ├── 01-tokenization.ipynb
│   │   ├── 02-tfidf-classifier.ipynb
│   │   └── 03-word2vec.ipynb
│   ├── slides/                     # Presentation slides
│   │   └── part01-slides.pdf
│   └── README.md                   # Part overview
│
├── part-02-rnn-to-transformers/    # Part 2: RNNs to Transformers
│   ├── theory/
│   │   ├── 01-rnn-basics.md
│   │   ├── 02-lstm-gru.md
│   │   ├── 03-attention.md
│   │   └── 04-transformers.md
│   ├── labs/
│   │   ├── lab02-rnn-models/
│   │   ├── lab03-transformers/
│   │   └── lab-exercises.md
│   ├── notebooks/
│   │   ├── 01-rnn-implementation.ipynb
│   │   ├── 02-lstm-sentiment.ipynb
│   │   ├── 03-seq2seq-translation.ipynb
│   │   └── 04-transformer-scratch.ipynb
│   ├── slides/
│   └── README.md
│
├── part-03-llms/                   # Part 3: Large Language Models
│   ├── theory/
│   │   ├── 01-gpt-evolution.md
│   │   ├── 02-scaling-laws.md
│   │   ├── 03-pretraining.md
│   │   └── 04-generation-strategies.md
│   ├── labs/
│   │   └── lab04-foundation-models/
│   ├── notebooks/
│   │   ├── 01-model-loading.ipynb
│   │   ├── 02-generation-params.ipynb
│   │   ├── 03-tokenizer-comparison.ipynb
│   │   └── 04-few-shot-learning.ipynb
│   ├── slides/
│   └── README.md
│
├── part-04-rag/                    # Part 4: RAG Systems
│   ├── theory/
│   │   ├── 01-rag-fundamentals.md
│   │   ├── 02-embeddings.md
│   │   ├── 03-vector-databases.md
│   │   └── 04-advanced-rag.md
│   ├── labs/
│   │   └── lab05-rag-pipeline/
│   ├── notebooks/
│   │   ├── 01-embeddings-basics.ipynb
│   │   ├── 02-vector-search.ipynb
│   │   ├── 03-chunking-strategies.ipynb
│   │   └── 04-complete-rag.ipynb
│   ├── slides/
│   └── README.md
│
├── part-05-finetuning/             # Part 5: Fine-Tuning
│   ├── theory/
│   │   ├── 01-sft-basics.md
│   │   ├── 02-peft-methods.md
│   │   ├── 03-lora-qlora.md
│   │   ├── 04-alignment.md
│   │   └── 05-evaluation.md
│   ├── labs/
│   │   ├── lab06-lora-finetuning/
│   │   ├── lab07-dpo-training/
│   │   └── lab08-evaluation/
│   ├── notebooks/
│   │   ├── 01-dataset-prep.ipynb
│   │   ├── 02-lora-training.ipynb
│   │   ├── 03-qlora-training.ipynb
│   │   ├── 04-dpo-alignment.ipynb
│   │   └── 05-model-evaluation.ipynb
│   ├── slides/
│   └── README.md
│
├── part-06-mcp/                    # Part 6: Model Context Protocol
│   ├── theory/
│   │   ├── 01-mcp-intro.md
│   │   ├── 02-architecture.md
│   │   └── 03-implementation.md
│   ├── labs/
│   │   ├── lab09-mcp-servers/
│   │   └── lab10-production-mcp/
│   ├── notebooks/
│   │   ├── 01-mcp-basics.ipynb
│   │   ├── 02-custom-server.ipynb
│   │   └── 03-client-integration.ipynb
│   ├── examples/
│   │   ├── filesystem-server/
│   │   ├── database-server/
│   │   └── api-server/
│   ├── slides/
│   └── README.md
│
├── part-07-agents/                 # Part 7: Agentic AI
│   ├── theory/
│   │   ├── 01-agent-foundations.md
│   │   ├── 02-reasoning-planning.md
│   │   ├── 03-langchain-langgraph.md
│   │   └── 04-multi-agent.md
│   ├── labs/
│   │   ├── lab11-langgraph-agents/
│   │   ├── lab12-multi-agent/
│   │   └── lab13-production-agents/
│   ├── notebooks/
│   │   ├── 01-simple-agent.ipynb
│   │   ├── 02-langgraph-workflow.ipynb
│   │   ├── 03-multi-agent-system.ipynb
│   │   └── 04-agent-deployment.ipynb
│   ├── examples/
│   │   ├── research-agent/
│   │   ├── coding-agent/
│   │   └── customer-service-agent/
│   ├── slides/
│   └── README.md
│
├── datasets/                       # Course datasets
│   ├── nlp-fundamentals/
│   │   ├── imdb-reviews/
│   │   └── ag-news/
│   ├── rag/
│   │   ├── arxiv-papers/
│   │   └── wiki-dumps/
│   ├── finetuning/
│   │   ├── alpaca/
│   │   ├── preference-data/
│   │   └── domain-specific/
│   └── README.md
│
├── resources/                      # Additional resources
│   ├── papers/                     # Research papers
│   │   ├── transformers/
│   │   ├── llms/
│   │   └── alignment/
│   ├── slides/                     # Course presentations
│   ├── references/                 # Reference materials
│   └── cheatsheets/               # Quick reference guides
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── models/                    # Model implementations
│   │   ├── __init__.py
│   │   ├── transformer.py
│   │   ├── rnn.py
│   │   └── attention.py
│   ├── data/                      # Data utilities
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── preprocessing.py
│   │   └── tokenizers.py
│   ├── training/                  # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   └── optimizers.py
│   ├── rag/                       # RAG components
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   ├── agents/                    # Agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── learning_assistant.py
│   │   └── code_reviewer.py
│   ├── evaluation/                # Evaluation utilities
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── benchmarks.py
│   └── utils/                     # General utilities
│       ├── __init__.py
│       ├── logging.py
│       ├── config.py
│       └── helpers.py
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_data.py
│   ├── test_training.py
│   ├── test_rag.py
│   └── test_agents.py
│
├── scripts/                       # Utility scripts
│   ├── download_models.py
│   ├── prepare_datasets.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── deploy_agent.py
│
├── configs/                       # Configuration files
│   ├── models/
│   │   ├── bert_config.yaml
│   │   ├── gpt_config.yaml
│   │   └── llama_config.yaml
│   ├── training/
│   │   ├── sft_config.yaml
│   │   ├── lora_config.yaml
│   │   └── dpo_config.yaml
│   ├── rag/
│   │   └── rag_config.yaml
│   └── agents/
│       └── agent_config.yaml
│
├── .github/                       # GitHub configuration
│   ├── workflows/
│   │   ├── tests.yml
│   │   ├── lint.yml
│   │   └── docs.yml
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
│
├── .vscode/                       # VS Code settings
│   ├── settings.json
│   ├── launch.json
│   └── extensions.json
│
├── docker/                        # Docker files
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
│
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
├── .pre-commit-config.yaml        # Pre-commit hooks
├── LICENSE                        # MIT License
├── Makefile                       # Build automation
├── README.md                      # Main README
├── plan.md                        # Course plan
├── pyproject.toml                 # Project metadata
├── requirements.txt               # Python dependencies
├── requirements-dev.txt           # Development dependencies
├── setup.py                       # Package setup
└── verify_setup.py               # Setup verification
```

## 📝 File Conventions

### Naming Conventions

**Python Files:**
- Modules: `lowercase_with_underscores.py`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

**Notebooks:**
- Format: `XX-descriptive-name.ipynb`
- Example: `01-tokenization-basics.ipynb`

**Markdown:**
- Format: `descriptive-name.md`
- Example: `transformer-architecture.md`

**Configs:**
- Format: `component_config.yaml`
- Example: `training_config.yaml`

### Directory Purposes

| Directory | Purpose |
|-----------|---------|
| `docs/` | Course documentation and guides |
| `part-*/` | Individual course modules |
| `theory/` | Theoretical explanations |
| `labs/` | Hands-on lab exercises |
| `notebooks/` | Jupyter notebooks |
| `slides/` | Presentation materials |
| `src/` | Reusable source code |
| `tests/` | Unit and integration tests |
| `scripts/` | Standalone utility scripts |
| `configs/` | Configuration files |
| `datasets/` | Course datasets |
| `resources/` | Additional learning materials |

## 🔧 Configuration Files

### .env.example
Template for environment variables

### .gitignore
Files to exclude from version control

### .pre-commit-config.yaml
Automated checks before commits

### pyproject.toml
Python project metadata and tool configs

### requirements.txt
Python package dependencies

### Makefile
Common commands automation

## 📦 Package Structure

```python
# Import structure
from src.models import TransformerModel
from src.data import DataLoader
from src.training import Trainer
from src.rag import RAGPipeline
from src.agents import LearningAssistant
```

## 🎯 Module Organization

### Part Structure

Each part follows this structure:

```
part-XX-name/
├── README.md          # Overview and objectives
├── theory/            # Conceptual explanations
├── labs/              # Practical exercises
├── notebooks/         # Interactive examples
└── slides/            # Presentations
```

### Lab Structure

```
lab-XX-name/
├── README.md          # Lab instructions
├── starter/           # Starter code
├── solution/          # Reference solution
├── data/              # Lab-specific data
└── tests/             # Lab tests
```

## 🚀 Quick Navigation

**For Students:**
- Start with [README.md](../README.md)
- Follow [setup.md](setup.md) for environment setup
- Begin with part-01-nlp-fundamentals/
- Use [commands.md](commands.md) for reference

**For Contributors:**
- Read [contributing.md](contributing.md)
- Follow [rules.md](rules.md) for conventions
- Check existing issues
- Submit PRs with tests

**For Instructors:**
- Review all part-*/theory/ content
- Test all labs and notebooks
- Update slides/ as needed
- Monitor student progress

## 📊 Size Guidelines

**Notebooks:**
- Theory: 10-15 cells
- Labs: 20-30 cells
- Include markdown explanations

**Code Files:**
- Max 300 lines per file
- Split large files into modules
- One class per file (generally)

**Documentation:**
- Tutorials: 1000-2000 words
- API docs: Complete docstrings
- Examples for all functions

## 🔍 Finding Content

### By Topic

```bash
# Find transformer content
find . -name "*transformer*"

# Find RAG content
find . -name "*rag*"

# Find agent content
find . -name "*agent*"
```

### By Type

```bash
# All notebooks
find . -name "*.ipynb"

# All theory
find . -path "*/theory/*.md"

# All labs
find . -path "*/labs/*"
```

## 📈 Metrics

**Current Stats:**
- 7 course parts
- 13 labs
- 40+ notebooks
- 30+ theory documents
- 100+ code examples

---

**💡 Tip:** Use the table of contents in each README.md for quick navigation within parts!
