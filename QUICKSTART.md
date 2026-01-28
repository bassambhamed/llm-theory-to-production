# Quick Start Guide

Get started with the Transformers & LLMs course in 5 minutes!

## ⚡ Fast Setup (5 minutes)

### 1. Clone or Download

```bash
# If you have git
git clone <repository-url>
cd transformers-llms-course

# Or download and extract the ZIP file
```

### 2. Setup Environment

**Option A: Using Make (Recommended)**
```bash
make setup
source venv/bin/activate
```

**Option B: Manual Setup**
```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# Install packages
pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
# Copy template
cp .env.example .env

# Edit .env and add your keys
# At minimum, add:
# - OPENAI_API_KEY or ANTHROPIC_API_KEY
# - HF_TOKEN (from huggingface.co)
```

### 4. Verify Installation

```bash
python verify_setup.py
```

### 5. Start Learning!

```bash
# Launch Jupyter
jupyter lab

# Or use make
make run-jupyter
```

## 🎯 Your First Session

### Open the first notebook:
`part-01-nlp-fundamentals/notebooks/01-introduction.ipynb`

### What you'll do:
1. ✅ Understand basic NLP concepts
2. ✅ Run your first tokenizer
3. ✅ Build a simple text classifier
4. ✅ Train word embeddings

## 📚 Course Navigation

### Beginner Path
```
Part 1 → Part 2 → Part 3 → Part 4
  ↓       ↓        ↓        ↓
 Labs   Labs     Labs    Labs
```

**Time:** 4-6 weeks (5-10 hours/week)

### Intermediate Path
```
Part 2 → Part 3 → Part 5
  ↓       ↓        ↓
 Labs   Labs     Labs
```

**Time:** 2-3 weeks (10-15 hours/week)

### Advanced Path
```
Part 5 → Part 6 → Part 7
  ↓       ↓        ↓
 Labs   Labs     Labs
```

**Time:** 2-3 weeks (15-20 hours/week)

## 🚀 Quick Commands

```bash
# Install everything
make install

# Run tests
make test

# Format code
make format

# Start Jupyter
make jupyter

# Clean cache
make clean

# See all commands
make help
```

## 📁 What's Where?

```
📦 Project Root
├── 📚 part-01-nlp-fundamentals/    ← Start here!
├── 📚 part-02-rnn-to-transformers/
├── 📚 part-03-llms/
├── 📚 part-04-rag/
├── 📚 part-05-finetuning/
├── 📚 part-06-mcp/
├── 📚 part-07-agents/
├── 📖 docs/                        ← Guides & references
├── 🗂️ datasets/                    ← Course datasets
└── 🔧 src/                         ← Reusable code
```

## 🆘 Common Issues

### Issue: ImportError
```bash
# Reinstall packages
pip install --upgrade -r requirements.txt
```

### Issue: CUDA out of memory
```python
# In notebooks, reduce batch size
batch_size = 8  # instead of 32
```

### Issue: API rate limit
```python
# Add delays between API calls
import time
time.sleep(1)
```

### Issue: Jupyter not working
```bash
# Restart Jupyter Lab
jupyter lab
```

## 💡 Pro Tips

1. **Save time:** Use `make` commands
2. **Stay organized:** Complete one part before moving to the next
3. **Practice:** Do all lab exercises
4. **Experiment:** Modify code and see what happens
5. **Ask questions:** Use GitHub Discussions

## 📞 Getting Help

- 📖 [Documentation](docs/)
- 🐛 [Report Issues](../../issues)
- 💬 [Discussions](../../discussions)
- 📧 Email: [your-email@example.com]

## 🎓 Learning Schedule

### Week 1-2: Fundamentals
- Part 1: NLP Fundamentals
- Part 2: RNN to Transformers
- **Goal:** Understand basics

### Week 3-4: LLMs & RAG
- Part 3: Large Language Models
- Part 4: RAG Systems
- **Goal:** Build applications

### Week 5-6: Advanced
- Part 5: Fine-tuning
- Part 6: MCP
- Part 7: Agents
- **Goal:** Production skills

## ✅ Daily Checklist

**Morning (1-2 hours):**
- [ ] Read theory section
- [ ] Watch related videos (if any)
- [ ] Take notes

**Afternoon (2-3 hours):**
- [ ] Work through labs
- [ ] Run code examples
- [ ] Experiment with parameters

**Evening (30 min):**
- [ ] Review what you learned
- [ ] Plan tomorrow's session
- [ ] Update progress tracker

## 🎯 Success Metrics

Track your progress:
- [ ] Complete 70% of labs
- [ ] Build 3 projects
- [ ] Pass final assessment
- [ ] Deploy 1 application

## 🔗 Useful Links

**Official Resources:**
- [Hugging Face Docs](https://huggingface.co/docs)
- [PyTorch Docs](https://pytorch.org/docs)
- [LangChain Docs](https://python.langchain.com)

**Community:**
- [Hugging Face Discord](https://huggingface.co/join/discord)
- [r/MachineLearning](https://reddit.com/r/MachineLearning)

**Papers:**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-3 Paper](https://arxiv.org/abs/2005.14165)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 🚦 Ready to Go?

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Start Jupyter
jupyter lab

# 3. Open first notebook
# part-01-nlp-fundamentals/notebooks/01-introduction.ipynb

# 4. Start learning! 🚀
```

---

**Questions?** Check [docs/setup.md](docs/setup.md) for detailed setup instructions.

**Happy Learning! 🎉**
