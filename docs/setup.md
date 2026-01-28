# Setup Guide

Complete setup instructions for the Transformers & LLMs course.

## 🎯 Overview

This guide will help you set up your development environment for all course modules.

## 📋 Prerequisites

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 16 GB
- Storage: 50 GB free
- OS: Linux, macOS, or Windows 10/11

**Recommended:**
- CPU: 8+ cores
- RAM: 32 GB
- GPU: NVIDIA GPU with 8+ GB VRAM
- Storage: 100 GB SSD

### Software Prerequisites

- **Conda** (Anaconda)
- **Git**
- **CUDA 11.8+** (for GPU support)

## 🚀 Installation

### Step 1: Install Conda

If you don't have Conda installed:

```bash
# Download Anaconda
# For Linux:
wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh
bash Anaconda3-latest-Linux-x86_64.sh

# For macOS (Apple Silicon):
wget https://repo.anaconda.com/archive/Anaconda3-latest-MacOSX-arm64.sh
bash Anaconda3-latest-MacOSX-arm64.sh

# For Windows: Download from
# https://www.anaconda.com/download
```

### Step 2: Clone Repository

```bash
git clone https://github.com/bassambhamed/llm-theory-to-production.git
cd llm-theory-to-production
```

### Step 3: Create Conda Environment

**Option A: Using environment.yml (Recommended)**

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate llm
```

**Option B: Manual setup**

```bash
# Create environment
conda create -n llm python=3.11 -y

# Activate environment
conda activate llm

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other packages
pip install -r requirements.txt
```

### Step 4: Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
nano .env  # or use your preferred editor
```

Add at minimum:
```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
HF_TOKEN=your_huggingface_token
```

### Step 5: Verify Installation

```bash
# Run verification script
python verify_setup.py
```

You should see ✓ marks for all core packages.

## 🧪 Testing Your Setup

### Test PyTorch and GPU

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Test Transformers

```python
from transformers import pipeline

# Test with a simple pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love this course!")
print(result)
```

## 📓 Jupyter Setup

Jupyter and JupyterLab are automatically installed when you create the environment from `environment.yml`.

To start Jupyter:

```bash
# Make sure environment is activated
conda activate llm

# Start Jupyter Lab
jupyter lab
```

## 🤗 Hugging Face Setup

### Authentication

```bash
# Login to Hugging Face
huggingface-cli login

# Or set token as environment variable
export HF_TOKEN=your_token_here
```

### Download Models

```bash
# Models will be cached automatically when you use them
# Cache location: ~/.cache/huggingface/

# Optional: Pre-download a model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"
```

## 🗄️ Vector Database Setup

### ChromaDB (Recommended for beginners)

ChromaDB works out of the box:

```python
import chromadb
client = chromadb.Client()
# That's it!
```

### Weaviate (Optional, for production)

```bash
# Start with Docker
docker run -d \
  -p 8080:8080 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  semitechnologies/weaviate:latest
```

## 🔧 Useful Commands

### Conda Environment Management

```bash
# List environments
conda env list

# Activate environment
conda activate llm

# Deactivate
conda deactivate

# Update environment
conda env update -f environment.yml

# Remove environment
conda env remove -n llm

# Export environment
conda env export > environment.yml
```

### Package Management

```bash
# Install single package
pip install package_name

# Update all packages
pip install --upgrade -r requirements.txt

# List installed packages
conda list
```

## 🆘 Troubleshooting

### Issue: CUDA out of memory

```python
# Reduce batch size in your code
batch_size = 8  # instead of 32

# Clear cache
import torch
torch.cuda.empty_cache()
```

### Issue: Package conflicts

```bash
# Recreate environment
conda deactivate
conda env remove -n llm
conda env create -f environment.yml
```

### Issue: Jupyter not working

```bash
# Reinstall Jupyter components
conda activate llm
conda install jupyter jupyterlab ipykernel -y
```

### Issue: Import errors

```bash
# Verify installation
python verify_setup.py

# Reinstall problematic package
pip install --upgrade --force-reinstall package_name
```

## ✅ Setup Checklist

Before starting the course, verify:

- [ ] Conda installed
- [ ] Environment created and activated
- [ ] All packages installed (run `verify_setup.py`)
- [ ] GPU working (if available)
- [ ] API keys configured in `.env`
- [ ] Jupyter Lab working
- [ ] First notebook runs successfully

## 🎓 Ready to Learn!

Once everything is setup:

```bash
# Activate environment
conda activate llm

# Start Jupyter
jupyter lab

# Open: part-01-nlp-fundamentals/notebooks/01-introduction.ipynb
```

## 📧 Getting Help

If you encounter issues:

1. Check this guide again
2. Run `python verify_setup.py`
3. Search [GitHub Issues](https://github.com/bassambhamed/llm-theory-to-production/issues)
4. Ask in [Discussions](https://github.com/bassambhamed/llm-theory-to-production/discussions)

---

**🎉 Setup complete! You're ready to dive into LLMs!**
