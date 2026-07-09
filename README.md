# Natural Language Processing Projects

Four end-to-end NLP systems implemented in Python and PyTorch: text classification, word embeddings, language modeling, and natural-language-to-SQL generation. Each project includes training/evaluation code, unit tests, reported results, and a project-level README with full setup and experiment details.

These were built as graduate coursework for the Natural Language Processing course at Cornell University and are intended to demonstrate practical ML engineering for Natural Langauge Processing: implementing models from scratch, comparing baselines, and measuring systems with task-appropriate metrics.

## Tech stack

Python · PyTorch · NumPy · Pandas · Hugging Face Transformers · NLTK · pytest · uv

## Projects summary

| Project | What it covers | Key techniques | Highlight |
| --- | --- | --- | --- |
| [Perceptrons and MLPs](./Perceptrons%20and%20MLPs/) | Text classification (SST-2, 20 Newsgroups) | Feature-based perceptron; token-embedding MLP | **MLP ≈75–67% test accuracy**; feature ablations and MPS batching study |
| [Word2Vec](./Word2Vec/) | Word embeddings + similarity | Skip-gram, negative sampling, subword vocab, WordNet retrofit | **0.395** Spearman on hidden test (2nd on course leaderboard) |
| [N-gram and Transformer](./N-gram%20and%20Transformer/) | Language modeling + ASR reranking | N-gram LMs; character-level Transformer; HUB WER | TF-Large: **3.34** test PPL, **6.46%** best HUB dev WER |
| [Natural Language to SQL Translator](./Natural%20Language%20to%20SQL%20Translator/) | NL → executable SQL (flights DB) | Fine-tuned / from-scratch T5; Gemma ICL | Fine-tuned T5: **0.65** Record F1 on test |

---

### 1. Perceptrons and MLPs

Classic vs. neural text classification on **SST-2** (sentiment) and **20 Newsgroups** (topics). A multiclass **perceptron** with hand-crafted features (bag-of-words, n-grams, lexicons, negation, subject lines) is compared to a from-scratch **MLP** that averages token embeddings and classifies through a small feed-forward network.

**Skills:** feature engineering, PyTorch training loops, early stopping, ablation analysis, inference batching on Apple MPS.

→ [Project README](./Perceptrons%20and%20MLPs/README.md)

### 2. Word2Vec

A **skip-gram Word2Vec** model with negative sampling, implemented from scratch, trained on large news corpora (dependency-parsed CoNLL + 1B-word shards). Embeddings are evaluated on isolated word-pair similarity (Spearman correlation). Subword (character n-gram) vocabularies and WordNet retrofitting were important for performance.

**Skills:** custom embedding training at scale, negative sampling, OOV handling, post-hoc embedding refinement, ranking metrics.

→ [Project README](./Word2Vec/README.md)

### 3. N-gram and Transformer

Side-by-side comparison of count-based **n-gram language models** (word / BPE / character; MLE, Laplace, linear interpolation) and a **decoder-only character-level Transformer**, evaluated on WSJ perplexity and HUB ASR *n*-best list reranking (WER).

**Skills:** probabilistic LMs, Transformer architecture from scratch, perplexity evaluation, speech-oriented reranking, hyperparameter scaling.

→ [Project README](./N-gram%20and%20Transformer/README.md)

### 4. Natural Language to SQL Translator

Maps natural-language questions about flights to **executable SQLite** over a flight database. Three approaches: **fine-tuning T5-small**, **training T5 from scratch**, and **few-shot in-context learning** with Gemma (TF-IDF example selection). Systems are scored with Query Exact Match and Record F1 (overlap of result sets after running predicted vs. gold SQL).

**Skills:** seq2seq fine-tuning, LLM prompting / ICL, schema-aware generation, execution-based evaluation.

→ [Project README](./Natural%20Language%20to%20SQL%20Translator/README.md)

---

## Getting started

### Prerequisites

- **Python 3.10** (required by all projects; the SQL project pins `>=3.10,<3.11`)
- [`uv`](https://github.com/astral-sh/uv) for dependency management (recommended)

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Set up any project

Dependencies are **not** shared at the repo root. `cd` into a project folder, sync, then activate (or use `uv run`):

```sh
cd "Perceptrons and MLPs"   # or Word2Vec / "N-gram and Transformer" / "Natural Language to SQL Translator"
uv sync
source .venv/bin/activate
```

Run unit tests from that same folder:

```sh
uv run pytest
```


Training-heavy projects (especially Word2Vec, Transformers, and T5) benefit from a Cloud GPU or Apple Silicon with MPS. Full CLI flags, data notes, and evaluation commands are in each project’s README.


