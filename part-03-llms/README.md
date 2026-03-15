# Part 3: From Transformers to Large Language Models

## 📚 Overview

Understand how transformers evolved into modern Large Language Models (LLMs), master prompt engineering techniques, and learn to leverage function calling to build tool-augmented applications.

## 🎯 Learning Objectives

By the end of this part, you will:

1. ✅ Trace the evolution from GPT-1 to GPT-4 and modern open-source LLMs
2. ✅ Understand scaling laws, emergent abilities, and multimodal models
3. ✅ Master pre-training concepts and decoding strategies
4. ✅ Apply prompt engineering techniques (zero-shot, few-shot, CoT, ToT)
5. ✅ Implement function calling and structured output extraction
6. ✅ Compare LLM capabilities across providers (OpenAI, Anthropic, open-source)

## 📖 Theory Topics

### 3.1 The Emergence of Large Language Models

- **The GPT Family Evolution**
  - GPT-1: Unsupervised pre-training + supervised fine-tuning
  - GPT-2: Scaling and zero-shot learning
  - GPT-3: Few-shot learning and in-context learning
  - GPT-4 / GPT-4o: Multimodal and advanced reasoning
- **Open-Source LLM Ecosystem**
  - LLaMA family (Meta), Mistral/Mixtral (Mistral AI)
  - Phi family (Microsoft), Gemma (Google), Qwen (Alibaba)
- **Scaling Laws**
  - Compute, data size, and parameters (Kaplan et al., Chinchilla)
  - Emergent abilities at scale
- **Multimodal LLMs**
  - Vision-Language Models (GPT-4V, LLaVA, Claude Vision)
  - Audio/speech integration (Whisper, GPT-4o audio)
- **Pre-training Pipeline**
  - Self-supervised learning on massive datasets
  - Data sources: Common Crawl, The Pile, RedPajama, FineWeb
- **Next-Token Prediction & Decoding Strategies**
  - Autoregressive generation, cross-entropy loss, perplexity
  - Greedy, temperature, Top-k, Top-p (nucleus sampling)
  - Repetition and frequency penalties
- **Context Window**
  - Sequence modeling limitations and extension techniques (RoPE scaling)

### 3.2 Prompt Engineering, In-Context Learning & Function Calling

- **Prompt Engineering Fundamentals**
  - Zero-shot and few-shot prompting
  - System prompts and role definition
  - Structured outputs (JSON mode)
- **Chain-of-Thought (CoT) Reasoning**
  - Step-by-step reasoning (zero-shot vs. few-shot CoT)
  - Self-Consistency: sampling multiple reasoning chains
- **Advanced Prompting Techniques**
  - Tree-of-Thoughts (ToT) and Graph-of-Thoughts
  - Prompt chaining and decomposition
- **Function Calling**
  - JSON schema tool definitions
  - Single and parallel tool calls
  - Function calling across providers (OpenAI, Anthropic, open-source)
  - Error handling and fallback strategies

## 🔬 Lab Exercises

### Lab 4: Interacting with Foundation Models

**Objectives:**
- Load and compare open-source models (Llama 3, Phi-3, Mistral)
- Generate text with different decoding strategies
- Analyze the impact of temperature and sampling parameters
- Visualize token probabilities and the generation process
- Compare tokenizers across different model families
- Test zero-shot and few-shot capabilities

**Tools:**
- Python, Hugging Face Transformers, OpenAI API, Anthropic API

### Lab 5: Prompt Engineering & Function Calling

**Objectives:**
- Implement zero-shot, few-shot, and CoT prompting strategies
- Compare prompt techniques across different models
- Build function calling workflows with tool definitions
- Implement structured output extraction (JSON mode)
- Evaluate prompt effectiveness on benchmark tasks

**Tools:**
- OpenAI API, Anthropic API, Hugging Face Transformers

## 📓 Notebooks

1. **lab4-foundation-models.ipynb** — Interacting with Foundation Models
2. **lab5-prompt-engineering.ipynb** — Prompt Engineering & Function Calling

## 🎞️ Slides

- **part3-llms.pdf**

## 🚀 Getting Started

```bash
# Activate environment
conda activate llm

# Start Jupyter
jupyter lab

# Open first notebook
# notebooks/lab4-foundation-models.ipynb
```

## 📋 Prerequisites

- Completion of Part 1 (NLP Fundamentals) and Part 2 (RNNs to Transformers)
- Basic understanding of the transformer architecture
- API keys for OpenAI and/or Anthropic (for API-based exercises)
- GPU recommended for running open-source models locally

## 📚 Resources

### Papers
- Radford et al. (2018) — GPT-1: Improving Language Understanding by Generative Pre-Training
- Radford et al. (2019) — GPT-2: Language Models are Unsupervised Multitask Learners
- Brown et al. (2020) — GPT-3: Language Models are Few-Shot Learners
- Kaplan et al. (2020) — Scaling Laws for Neural Language Models
- Hoffmann et al. (2022) — Chinchilla: Training Compute-Optimal Large Language Models
- Wei et al. (2022) — Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- Yao et al. (2023) — Tree of Thoughts: Deliberate Problem Solving with Large Language Models

### Tutorials
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com)

### Datasets
- MMLU (benchmark evaluation)
- HellaSwag (common-sense reasoning)
- HumanEval (code generation)

## 🎓 Assessment

To complete this part:
- [ ] Complete Lab 4 — Interacting with Foundation Models
- [ ] Complete Lab 5 — Prompt Engineering & Function Calling
- [ ] Understand scaling laws and emergent abilities
- [ ] Compare prompting strategies and their trade-offs
- [ ] Build a function calling workflow end-to-end

---

**Previous:** [Part 2: From RNNs to Transformers](../part-02-rnn-to-transformers/) · **Next:** [Part 4: Retrieval-Augmented Generation (RAG)](../part-04-rag/)
