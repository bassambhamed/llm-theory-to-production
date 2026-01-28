# Datasets

This directory contains datasets used throughout the course.

## 📊 Dataset Organization

```
datasets/
├── nlp-fundamentals/    # Classical NLP datasets
├── rag/                 # Documents for RAG systems
└── finetuning/          # Instruction tuning datasets
```

## 📦 Available Datasets

### NLP Fundamentals

**IMDB Reviews**
- Task: Sentiment analysis
- Size: 50,000 reviews
- Download: Automatically via Hugging Face Datasets

**AG News**
- Task: Text classification
- Size: 120,000 news articles (4 categories)
- Download: Automatically via Hugging Face Datasets

**CoNLL 2003**
- Task: Named Entity Recognition
- Download: Automatically via Hugging Face Datasets

### RAG Documents

**ArXiv Papers**
- Scientific papers for RAG testing
- Format: PDF, plain text
- Use: Document retrieval practice

**Wikipedia Dumps**
- Sample Wikipedia articles
- Format: Markdown, plain text
- Use: Knowledge base for RAG

### Fine-Tuning

**Alpaca Dataset**
- Instruction-following examples
- Size: 52,000 samples
- Download: `huggingface-cli download tatsu-lab/alpaca`

**Dolly Dataset**
- Human-generated instruction dataset
- Size: 15,000 samples
- Download: Automatically via Hugging Face

## 🚀 Usage

### Loading Datasets with Hugging Face

```python
from datasets import load_dataset

# Load IMDB
dataset = load_dataset("imdb")

# Load AG News
dataset = load_dataset("ag_news")

# Load Alpaca
dataset = load_dataset("tatsu-lab/alpaca")
```

### Manual Download

Some datasets need manual download:

```bash
# Download ArXiv papers (example)
mkdir -p datasets/rag/arxiv-papers
cd datasets/rag/arxiv-papers
# Add your PDF files here
```

## 📏 Dataset Statistics

| Dataset | Task | Samples | Size | License |
|---------|------|---------|------|---------|
| IMDB | Sentiment | 50K | ~80 MB | Apache 2.0 |
| AG News | Classification | 120K | ~30 MB | MIT |
| Alpaca | Instruction | 52K | ~20 MB | CC BY-NC |
| Dolly | Instruction | 15K | ~10 MB | CC BY-SA |

## 🔍 Data Preprocessing

### Text Cleaning

```python
import re

def clean_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text
```

### Tokenization

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer(text, padding=True, truncation=True)
```

## ⚠️ Important Notes

- **Large files are in .gitignore** - Download them locally
- **API keys required** - Some datasets need Hugging Face authentication
- **Storage requirements** - Ensure 50+ GB free space
- **License compliance** - Check dataset licenses for your use case

## 📚 Adding Your Own Datasets

To add custom datasets:

1. Create subdirectory: `datasets/your-dataset/`
2. Add data files
3. Create `README.md` describing the dataset
4. Update this file with dataset info

## 🤝 Contributing

To contribute datasets:

1. Ensure proper licensing
2. Include preprocessing scripts
3. Document format and structure
4. Provide usage examples

---

**For questions or dataset requests, open an issue on GitHub.**
