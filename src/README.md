# Source Code

Reusable Python modules and utilities for the course.

## 📦 Package Structure

```
src/
├── models/          # Model implementations
├── data/            # Data loading and processing
├── training/        # Training utilities
├── rag/             # RAG components
├── agents/          # Agent implementations
├── evaluation/      # Evaluation metrics
└── utils/           # General utilities
```

## 🔧 Modules

### models/
Transformer and LLM model implementations

```python
from src.models import TransformerModel

model = TransformerModel(vocab_size=10000, d_model=512)
```

### data/
Data loaders and preprocessing

```python
from src.data import TextDataLoader

loader = TextDataLoader("data.txt", batch_size=32)
```

### training/
Training loops and utilities

```python
from src.training import Trainer

trainer = Trainer(model, train_data, val_data)
trainer.train(epochs=10)
```

### rag/
RAG pipeline components

```python
from src.rag import RAGPipeline

rag = RAGPipeline(embedder, retriever, generator)
answer = rag.query("What is attention mechanism?")
```

### agents/
AI agent implementations

```python
from src.agents import LearningAssistant

assistant = LearningAssistant()
response = assistant.explain("What is LoRA?")
```

### evaluation/
Metrics and benchmarking

```python
from src.evaluation import evaluate_model

metrics = evaluate_model(model, test_data)
```

### utils/
Helper functions

```python
from src.utils import setup_logging, load_config

logger = setup_logging()
config = load_config("config.yaml")
```

## 🚀 Usage

### Installation

The `src` directory is automatically in the Python path when you run notebooks from the project root.

For standalone scripts:

```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

### Importing Modules

```python
# From notebooks
from src.models import TransformerModel
from src.data import TextDataLoader

# Within src/
from .models import TransformerModel
from .utils import setup_logging
```

## 📝 Code Style

All code follows:
- **PEP 8** style guide
- **Type hints** for functions
- **Google-style** docstrings
- **Black** formatting (line length 88)

Example:

```python
from typing import List, Optional
import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """Transformer model implementation.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers

    Example:
        >>> model = TransformerModel(vocab_size=10000, d_model=512)
        >>> output = model(input_ids)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6
    ):
        super().__init__()
        self.vocab_size = vocab_size
        # ... implementation

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        # ... implementation
        return output
```

## 🧪 Testing

Tests are in the `tests/` directory:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_models.py

# With coverage
pytest --cov=src tests/
```

## 📚 Documentation

Generate API documentation:

```bash
# Using pdoc
pdoc --html src -o docs/api

# Open docs/api/src/index.html
```

## 🤝 Contributing

To add new modules:

1. Create module in appropriate subdirectory
2. Add type hints and docstrings
3. Write unit tests
4. Update this README

## 🔍 Code Organization

### When to add code here vs notebooks

**Add to `src/`:**
- Reusable functions and classes
- Complex algorithms
- Production-ready code
- Code used across multiple notebooks

**Keep in notebooks:**
- Exploratory analysis
- Visualizations
- Step-by-step tutorials
- One-off experiments

---

**For questions about code organization, see [docs/rules.md](../docs/rules.md)**
