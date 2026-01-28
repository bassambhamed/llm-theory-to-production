# Project Rules & Best Practices

Guidelines and conventions for contributing to and using this course repository.

## 📜 Code of Conduct

### Core Principles

1. **Be Respectful:** Treat everyone with respect and kindness
2. **Be Inclusive:** Welcome learners of all backgrounds
3. **Be Collaborative:** Share knowledge and help others
4. **Be Professional:** Maintain professionalism in all interactions
5. **Be Patient:** Remember everyone learns at their own pace

### Expected Behavior

✅ Ask questions freely
✅ Share helpful resources
✅ Provide constructive feedback
✅ Acknowledge others' contributions
✅ Report issues responsibly

### Unacceptable Behavior

❌ Harassment or discrimination
❌ Sharing solutions to assignments
❌ Plagiarism
❌ Spamming or trolling
❌ Disclosing private information

## 💻 Coding Standards

### Python Style Guide

Follow [PEP 8](https://pep8.org/) with these specifics:

```python
# Line length: 88 characters (Black default)
# Indentation: 4 spaces
# Quotes: Double quotes for strings

# Good
def train_model(model: nn.Module, data: DataLoader) -> dict:
    """Train the model on given data.

    Args:
        model: PyTorch model to train
        data: DataLoader with training data

    Returns:
        Dictionary with training metrics
    """
    metrics = {}
    # Training logic here
    return metrics

# Bad
def train(m,d):
    metrics={}
    return metrics
```

### Naming Conventions

```python
# Classes: PascalCase
class TransformerModel:
    pass

# Functions and variables: snake_case
def calculate_attention_scores():
    num_heads = 8
    return scores

# Constants: UPPER_SNAKE_CASE
MAX_SEQUENCE_LENGTH = 512
LEARNING_RATE = 1e-4

# Private methods: _leading_underscore
def _internal_helper():
    pass
```

### Type Hints

Always use type hints for function parameters and returns:

```python
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn

def forward(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Forward pass."""
    outputs = {}
    return logits, outputs
```

### Docstrings

Use Google-style docstrings:

```python
def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """Compute loss between predictions and targets.

    Args:
        predictions: Model predictions of shape (batch_size, num_classes)
        targets: Ground truth labels of shape (batch_size,)
        reduction: Type of reduction to apply ('none', 'mean', 'sum')

    Returns:
        Computed loss as a scalar tensor

    Raises:
        ValueError: If predictions and targets shapes don't match

    Example:
        >>> predictions = torch.randn(32, 10)
        >>> targets = torch.randint(0, 10, (32,))
        >>> loss = compute_loss(predictions, targets)
    """
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError("Batch sizes don't match")
    return F.cross_entropy(predictions, targets, reduction=reduction)
```

## 📁 File Organization

### Directory Structure

```
project/
├── src/                    # Source code
│   ├── __init__.py
│   ├── models/            # Model implementations
│   ├── data/              # Data loading and processing
│   ├── training/          # Training scripts
│   └── utils/             # Utility functions
├── tests/                 # Unit tests
│   ├── test_models.py
│   └── test_data.py
├── notebooks/             # Jupyter notebooks
│   ├── 01_exploration.ipynb
│   └── 02_analysis.ipynb
├── configs/               # Configuration files
│   └── model_config.yaml
├── scripts/               # Standalone scripts
│   ├── train.py
│   └── evaluate.py
├── docs/                  # Documentation
├── requirements.txt       # Dependencies
└── README.md
```

### File Naming

```
# Python modules: lowercase with underscores
data_loader.py
model_utils.py

# Notebooks: numbered prefix, descriptive name
01_data_exploration.ipynb
02_model_training.ipynb
03_results_analysis.ipynb

# Configs: descriptive names
bert_config.yaml
training_config.json

# Scripts: action verbs
train_model.py
evaluate_results.py
preprocess_data.py
```

## 🧪 Testing Standards

### Test Coverage

- Minimum 80% code coverage
- Test all public APIs
- Include edge cases
- Test error handling

### Test Structure

```python
import pytest
import torch
from src.models import TransformerModel

class TestTransformerModel:
    """Test suite for TransformerModel."""

    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        return TransformerModel(vocab_size=1000, d_model=512)

    def test_forward_shape(self, model):
        """Test output shape is correct."""
        batch_size, seq_len = 4, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        output = model(input_ids)
        assert output.shape == (batch_size, seq_len, 1000)

    def test_invalid_input_raises_error(self, model):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError):
            model(torch.randn(4, 32, 512))  # Wrong shape

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_training_step(self, model):
        """Test training step runs without errors."""
        # Training test logic
        pass
```

### Test Markers

```python
# Mark slow tests
@pytest.mark.slow

# Mark GPU-required tests
@pytest.mark.gpu

# Mark integration tests
@pytest.mark.integration

# Skip test conditionally
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
```

## 📊 Experiment Tracking

### Weights & Biases Integration

```python
import wandb

# Initialize run
wandb.init(
    project="llm-course",
    name="experiment-1",
    config={
        "learning_rate": 1e-4,
        "batch_size": 32,
        "model": "transformer"
    }
)

# Log metrics
wandb.log({
    "train_loss": loss.item(),
    "train_accuracy": accuracy,
    "epoch": epoch
})

# Log artifacts
wandb.save("model.pt")

# Finish run
wandb.finish()
```

### Configuration Management

Use YAML for configs:

```yaml
# configs/training_config.yaml
model:
  name: "transformer"
  d_model: 512
  num_heads: 8
  num_layers: 6

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 10
  warmup_steps: 1000

data:
  max_length: 512
  train_path: "data/train.jsonl"
  val_path: "data/val.jsonl"
```

## 🔐 Security & Privacy

### API Keys Management

```bash
# Never commit API keys!
# Use .env file (add to .gitignore)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...

# Load in code
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

### Data Privacy

- Never commit personal data
- Anonymize datasets
- Use synthetic data for examples
- Follow GDPR/privacy regulations

### Model Security

- Validate all inputs
- Sanitize outputs
- Implement rate limiting
- Monitor for abuse

## 📝 Documentation Standards

### README Structure

Every module should have a README with:

1. Overview
2. Installation
3. Usage examples
4. API documentation
5. Contributing guidelines

### Code Comments

```python
# Use comments for WHY, not WHAT

# Good: Explains reasoning
# Using cosine scheduler because it prevents sudden drops
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Bad: States the obvious
# Create scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Use TODO for future work
# TODO(username): Add support for multi-GPU training

# Use FIXME for known issues
# FIXME: This fails for empty sequences

# Use NOTE for important information
# NOTE: Must be called before optimizer.step()
```

## 🚀 Deployment Guidelines

### Model Deployment Checklist

- [ ] Model is quantized for efficiency
- [ ] Inference latency is acceptable
- [ ] Memory usage is within limits
- [ ] Error handling is robust
- [ ] Monitoring is in place
- [ ] API documentation is complete
- [ ] Security measures are implemented
- [ ] Load testing is done

### Production Code Requirements

```python
# Include error handling
try:
    output = model.generate(input_ids)
except torch.cuda.OutOfMemoryError:
    logger.error("OOM error, reducing batch size")
    output = model.generate(input_ids[:len(input_ids)//2])

# Add logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"Processing {len(batch)} examples")

# Validate inputs
def validate_input(text: str) -> str:
    """Validate and sanitize input text."""
    if not text or len(text) > MAX_LENGTH:
        raise ValueError(f"Text length must be 1-{MAX_LENGTH}")
    return text.strip()

# Add type checking at runtime
from typing import runtime_checkable
```

## 📦 Version Control

### Git Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: <type>(<scope>): <description>

# Types:
feat: Add new feature
fix: Fix bug
docs: Update documentation
style: Format code
refactor: Refactor code
test: Add tests
chore: Maintenance tasks

# Examples:
git commit -m "feat(transformer): add rotary position embeddings"
git commit -m "fix(data): handle empty sequences correctly"
git commit -m "docs(readme): add installation instructions"
```

### Branch Naming

```bash
# Format: <type>/<description>

feature/add-rag-pipeline
bugfix/fix-attention-mask
docs/update-readme
experiment/try-different-optimizer
```

### Pull Request Guidelines

1. **Title:** Clear, descriptive
2. **Description:** What, why, how
3. **Tests:** All tests pass
4. **Documentation:** Updated if needed
5. **Review:** Request at least one review

## 🎯 Performance Optimization

### Training Optimization

```python
# Use mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input_ids)
    loss = criterion(output, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Inference Optimization

```python
# Use torch.no_grad() for inference
with torch.no_grad():
    output = model(input_ids)

# Use eval mode
model.eval()

# Use torch.inference_mode() (faster than no_grad)
with torch.inference_mode():
    output = model(input_ids)
```

## ⚠️ Common Pitfalls to Avoid

1. **Not setting random seeds** → Non-reproducible results
2. **Forgetting model.eval()** → Wrong behavior during inference
3. **Not handling CUDA OOM** → Crashes
4. **Hardcoded paths** → Breaks on different systems
5. **No input validation** → Security vulnerabilities
6. **Ignoring type hints** → Runtime errors
7. **Missing documentation** → Confusion
8. **Not using version control** → Lost work

## ✅ Quality Checklist

Before submitting code:

- [ ] Code follows style guide
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] No hardcoded values
- [ ] Error handling is present
- [ ] Type hints are added
- [ ] Comments explain complex logic
- [ ] No commented-out code
- [ ] No print() statements (use logging)
- [ ] Secrets are not committed

---

**💡 Remember:** Good code is readable, maintainable, and well-documented!
