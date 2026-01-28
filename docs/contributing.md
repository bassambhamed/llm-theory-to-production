# Contributing Guide

Thank you for your interest in contributing to the Transformers & LLMs course! This guide will help you get started.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **Course Content**
   - Improve explanations and theory
   - Add practical examples
   - Create new labs and exercises
   - Update outdated information

2. **Code Improvements**
   - Fix bugs
   - Improve performance
   - Add features
   - Refactor code

3. **Documentation**
   - Fix typos and grammar
   - Improve clarity
   - Add examples
   - Translate content

4. **Issue Reports**
   - Bug reports
   - Feature requests
   - Documentation issues
   - Questions

## 🚀 Getting Started

### 1. Fork the Repository

```bash
# Click "Fork" on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/transformers-llms-course.git
cd transformers-llms-course
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or bugfix branch
git checkout -b bugfix/issue-description
```

## 📝 Contribution Workflow

### For Course Content

1. **Identify what to improve**
   - Review existing content
   - Check open issues
   - Discuss in Discussions

2. **Make changes**
   - Follow course structure
   - Maintain consistent style
   - Add practical examples
   - Test all code

3. **Submit PR**
   - Describe changes clearly
   - Link related issues
   - Add screenshots if relevant

### For Code Changes

1. **Write clean code**
   - Follow PEP 8
   - Add type hints
   - Write docstrings
   - Keep functions small

2. **Add tests**
   ```bash
   # Run tests
   pytest tests/

   # Check coverage
   pytest --cov=src tests/
   ```

3. **Format and lint**
   ```bash
   # Format code
   black src/ tests/
   isort src/ tests/

   # Lint code
   ruff check src/ tests/
   mypy src/
   ```

## 📋 Pull Request Process

### Before Submitting

✅ Tests pass
✅ Code is formatted
✅ Documentation updated
✅ No merge conflicts
✅ Commit messages follow convention

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Motivation
Why is this change needed?

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
How was this tested?

## Screenshots (if applicable)
Add screenshots here

## Checklist
- [ ] Tests pass
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] Self-review completed
```

### Review Process

1. Maintainers will review within 2-3 days
2. Address review comments
3. Request re-review
4. Merge when approved

## 🎨 Style Guidelines

### Python Code

```python
# Good
def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 10
) -> dict[str, float]:
    """Train the model on provided data.

    Args:
        model: PyTorch model to train
        data_loader: DataLoader with training data
        optimizer: Optimizer for training
        num_epochs: Number of training epochs

    Returns:
        Dictionary containing training metrics
    """
    metrics = {"loss": [], "accuracy": []}

    for epoch in range(num_epochs):
        # Training logic
        pass

    return metrics
```

### Documentation

```markdown
# Use clear headings

## Second level

### Third level

- Use bullet points for lists
- Keep sentences concise
- Include code examples

```python
# Always include code blocks with language
def example():
    pass
```

**Bold** for emphasis
*Italic* for technical terms
`code` for inline code
```

### Notebooks

```python
# Cell 1: Imports
import torch
import transformers

# Cell 2: Setup
model = transformers.AutoModel.from_pretrained("bert-base-uncased")

# Cell 3: Main logic
# Clear explanations in markdown cells
```

## 📚 Content Guidelines

### Theory Sections

1. **Start with motivation**
   - Why is this important?
   - What problem does it solve?

2. **Explain concepts clearly**
   - Use simple language
   - Provide intuition
   - Add diagrams

3. **Include examples**
   - Real-world applications
   - Code demonstrations
   - Visual aids

4. **End with summary**
   - Key takeaways
   - Further reading
   - Practice exercises

### Lab Sections

1. **Clear objectives**
   - What will students learn?
   - What will they build?

2. **Step-by-step instructions**
   - Numbered steps
   - Expected outputs
   - Troubleshooting tips

3. **Complete code**
   - Well-commented
   - Runnable examples
   - Error handling

4. **Exercises**
   - Varying difficulty
   - Clear requirements
   - Hints and solutions

## 🐛 Reporting Issues

### Bug Reports

Include:
- **Description:** What happened?
- **Expected:** What should happen?
- **Steps to reproduce**
- **Environment:** OS, Python version, etc.
- **Error messages:** Full traceback

Example:

```markdown
## Bug Description
Model training fails with CUDA OOM error

## Expected Behavior
Model should train successfully

## Steps to Reproduce
1. Run `python train.py`
2. Use batch_size=32
3. GPU has 8GB VRAM

## Environment
- OS: Ubuntu 22.04
- Python: 3.11
- PyTorch: 2.1.0
- GPU: NVIDIA RTX 3070

## Error Message
```
RuntimeError: CUDA out of memory
```
```

### Feature Requests

Include:
- **Problem:** What problem does this solve?
- **Solution:** Proposed implementation
- **Alternatives:** Other approaches considered
- **Benefits:** Why add this feature?

## 🧪 Testing Guidelines

### Writing Tests

```python
import pytest
import torch
from src.models import TransformerModel

class TestTransformerModel:
    """Test TransformerModel class."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        return TransformerModel(
            vocab_size=1000,
            d_model=512,
            num_heads=8
        )

    def test_forward_pass(self, model):
        """Test forward pass produces correct output."""
        batch_size, seq_len = 4, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        output = model(input_ids)

        assert output.shape == (batch_size, seq_len, 1000)
        assert not torch.isnan(output).any()

    def test_invalid_input(self, model):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError):
            model(torch.randn(4, 32))  # Wrong dtype
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run only fast tests
pytest -m fast

# Run with verbose output
pytest -v
```

## 📊 Performance Guidelines

### Code Optimization

```python
# Use appropriate data types
tensor = torch.tensor([1, 2, 3], dtype=torch.float16)  # For mixed precision

# Avoid unnecessary computations
with torch.no_grad():  # For inference
    output = model(input_ids)

# Use vectorization
# Bad
for i in range(len(data)):
    result[i] = data[i] * 2

# Good
result = data * 2

# Profile code
import cProfile
cProfile.run('train_model()')
```

### Memory Management

```python
# Clear cache when needed
torch.cuda.empty_cache()

# Use gradient accumulation
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Delete large objects
del large_tensor
torch.cuda.empty_cache()
```

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

Top contributors may become maintainers!

## ❓ Questions?

- **Technical questions:** Open an issue
- **General discussion:** GitHub Discussions
- **Security issues:** Email maintainers directly
- **Other:** Contact course instructors

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Every contribution, no matter how small, helps improve the course for everyone. We appreciate your time and effort!

---

**Happy Contributing! 🎉**
