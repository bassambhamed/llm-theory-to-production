# Part 2: From RNNs to Transformers

## 📚 Overview

Bridge the gap between classical NLP and modern language models by mastering recurrent neural networks (RNNs/LSTMs), understanding their limitations, and discovering how the Transformer architecture revolutionized sequence modeling. Fine-tune pre-trained models (BERT, T5) and compare them to RNN baselines on real-world tasks.

## 🎯 Learning Objectives

By the end of this part, you will:

1. ✅ Implement text classification and sequence-to-sequence models with LSTMs
2. ✅ Understand RNN limitations (vanishing gradients, information bottleneck)
3. ✅ Master the Transformer architecture (self-attention, multi-head attention, positional encoding)
4. ✅ Fine-tune BERT for sentiment analysis and compare with LSTM baselines
5. ✅ Use T5 for multi-task sequence-to-sequence generation (summarization, translation)
6. ✅ Visualize attention weights and interpret Transformer behavior
7. ✅ Compare tokenization strategies (Word-level, BPE, WordPiece, SentencePiece)

## 📖 Theory Topics

### 2.1 Recurrent Neural Networks and LSTMs

- **LSTM Architecture**
  - Input, forget, and output gates
  - Cell state vs. hidden state
  - Gradient flow through time
- **Sentiment Classification with LSTM**
  - Custom tokenizer and vocabulary construction
  - Embedding layer, bidirectional LSTM, classification head
  - Variable-length sequence handling (padding and packing)
- **Sequence-to-Sequence Models**
  - Encoder-Decoder architecture with attention
  - Teacher forcing and beam search
  - English-to-French number translation
- **RNN Limitations**
  - Vanishing and exploding gradients (empirical demonstration)
  - Information bottleneck on long sequences
  - Sequential computation (no parallelism)

### 2.2 The Transformer Architecture

- **Core Mechanisms**
  - Scaled Dot-Product Attention: `Attention(Q,K,V) = softmax(QK^T / √d_k) V`
  - Multi-Head Attention: `MultiHead(Q,K,V) = Concat(h₁,...,h_H) W^O`
  - Position-wise Feed-Forward Networks
  - Residual connections and Layer Normalization
  - Sinusoidal Positional Encoding
- **BERT (Encoder-Only)**
  - Bidirectional Encoder Representations from Transformers
  - Pre-training: Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)
  - Fine-tuning strategy: small learning rate, few epochs, warmup scheduling
  - [CLS] token for classification tasks
- **T5 (Encoder-Decoder)**
  - Text-to-Text Transfer Transformer
  - Unified task format with prefix instructions
  - Multi-task capabilities: summarization, translation, classification, grammar checking
- **Attention Visualization**
  - Extracting attention weights across layers and heads
  - Attention patterns: diagonal, vertical stripe, distributed
  - [CLS] token attention evolution across layers
  - Negation handling and compositional semantics
- **Tokenization Strategies**
  - Word-level vs. subword tokenization
  - BPE (GPT-2, RoBERTa), WordPiece (BERT), SentencePiece/Unigram (T5, XLNet)
  - Vocabulary size vs. sequence length trade-off
  - Training custom BPE tokenizers
  - Impact on downstream performance and OOV handling

## 🔬 Lab Exercises

### Lab 2: Building RNN-Based Models

**Objectives:**
- Implement LSTM-based sentiment classification on IMDB reviews
- Build an Encoder-Decoder Seq2Seq model for translation
- Visualize LSTM hidden states across time steps
- Demonstrate vanishing gradients empirically and analyze gradient flow

**Tools:**
- Python, PyTorch, Hugging Face Datasets, Matplotlib

### Lab 3: Transformer for Classification and Seq2Seq

**Objectives:**
- Fine-tune BERT for sentiment analysis on IMDB (compare to LSTM baseline from Lab 2)
- Use T5 for summarization and translation tasks
- Compare Transformer vs. LSTM performance on the same classification task
- Visualize attention weights across layers and heads
- Experiment with tokenizers (BPE, WordPiece, SentencePiece) and measure impact

**Tools:**
- Python, Hugging Face Transformers, Tokenizers library

## 📓 Notebooks

1. **lab2-rnn-based-models.ipynb** — Building RNN-Based Models (LSTM classification, Seq2Seq, gradient analysis)
2. **lab3-transformer-classification-seq2seq.ipynb** — Transformer for Classification and Seq2Seq (BERT fine-tuning, T5, attention visualization, tokenizer comparison)

## 🎞️ Slides

- **part2-rnn-to-transformers.pdf** — Main course slides
- **AttentionIsAllYouNeed.pdf** — Original Transformer paper slides
- **tutoriel_transformer.pdf** — Transformer tutorial

## 🚀 Getting Started

```bash
# Activate environment
conda activate llm

# Start Jupyter
jupyter lab

# Open first notebook
# notebooks/lab2-rnn-based-models.ipynb
```

## 📋 Prerequisites

- Completion of Part 1 (NLP Fundamentals)
- Basic Python and PyTorch (tensors, nn.Module)
- Understanding of word embeddings and text classification
- GPU recommended for BERT fine-tuning

## 📚 Resources

### Papers
- Hochreiter & Schmidhuber (1997) — Long Short-Term Memory
- Bahdanau et al. (2015) — Neural Machine Translation by Jointly Learning to Align and Translate
- Vaswani et al. (2017) — Attention Is All You Need
- Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- Raffel et al. (2020) — T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- Sennrich et al. (2016) — Neural Machine Translation of Rare Words with Subword Units (BPE)

### Tutorials
- [PyTorch RNN Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers)

### Datasets
- IMDB Reviews (sentiment analysis — shared with Part 1 for fair comparison)
- English-to-French number pairs (Seq2Seq translation)

## 🎓 Assessment

To complete this part:
- [ ] Complete Lab 2 — Building RNN-Based Models
- [ ] Complete Lab 3 — Transformer for Classification and Seq2Seq
- [ ] Understand LSTM gates and gradient flow
- [ ] Compare LSTM vs. BERT performance on IMDB sentiment analysis
- [ ] Visualize and interpret Transformer attention patterns
- [ ] Compare tokenization strategies and their trade-offs

---

**Previous:** [Part 1: NLP Fundamentals](../part-01-nlp-fundamentals/) · **Next:** [Part 3: From Transformers to LLMs](../part-03-llms/)
