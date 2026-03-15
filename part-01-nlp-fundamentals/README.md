# Part 1: NLP Fundamentals

## 📚 Overview

Introduction to Natural Language Processing, covering classical techniques and establishing the foundation for modern deep learning approaches.

## 🎯 Learning Objectives

By the end of this part, you will:

1. ✅ Understand core NLP concepts and challenges
2. ✅ Master classical NLP techniques (N-grams, TF-IDF, Word2Vec)
3. ✅ Build text classification models
4. ✅ Evaluate NLP model performance
5. ✅ Compare classical vs modern approaches

## 📖 Theory Topics

### 1. Introduction to NLP
- What is Natural Language Processing?
- Core NLP tasks and challenges
- Historical evolution of language models

### 2. Text Representation
- Tokenization fundamentals
- Bag-of-Words and TF-IDF
- Word embeddings (Word2Vec, GloVe, FastText)

### 3. Evaluation Metrics
- Classification metrics (Precision, Recall, F1)
- Perplexity for language models
- BLEU and ROUGE for generation

## 🔬 Lab Exercises

### Lab 1: Classical NLP Techniques

**Objectives:**
- Text preprocessing and tokenization
- Build TF-IDF text classifier
- Train Word2Vec embeddings
- Explore word similarities

**Tools:**
- Python, NLTK, scikit-learn, Gensim


## 📓 Notebooks

1. **01-introduction.ipynb** - NLP basics and motivation
2. **02-tokenization.ipynb** - Text preprocessing
3. **03-tfidf-classifier.ipynb** - Build classifier
4. **04-word2vec.ipynb** - Train word embeddings
5. **05-evaluation.ipynb** - Metrics and evaluation

## 🚀 Getting Started

```bash
# Activate environment
conda activate llm

# Start Jupyter
jupyter lab

# Open first notebook
# notebooks/01-introduction.ipynb
```

## 📋 Prerequisites

- Basic Python programming
- Understanding of machine learning concepts
- Linear algebra fundamentals (vectors, matrices)

## 📚 Resources

### Papers
- Mikolov et al. (2013) - Word2Vec
- Pennington et al. (2014) - GloVe

### Tutorials
- [NLTK Documentation](https://www.nltk.org/)
- [Gensim Word2Vec Tutorial](https://radimrehurek.com/gensim/)

### Datasets
- IMDB Reviews (sentiment analysis)
- AG News (text classification)
- Sample corpus for Word2Vec training

## 🎓 Assessment

To complete this part:
- [ ] Complete Lab
- [ ] Understand evaluation metrics
- [ ] Compare different embedding methods

---

**Next:** [Part 2: From RNNs to Transformers](../part-02-rnn-to-transformers/)
