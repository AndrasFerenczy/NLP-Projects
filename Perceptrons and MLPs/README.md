# Text Classification with Perceptrons and MLPs

This project explores classic and neural approaches to text classification. I built a feature-based **perceptron** and a token-embedding **multi-layer perceptron (MLP)** from scratch, and evaluated both on two standard benchmarks: **SST-2** (binary sentiment) and **20 Newsgroups** (topic classification).

The perceptron uses hand-crafted features such as bag-of-words, n-grams, subject-line tokens, sentiment lexicons, and negation markers. The MLP embeds tokens, averages them into a document representation, and classifies through a small feed-forward network with dropout. The goal was to compare how far careful feature engineering can go against a simple neural baseline, and to measure how batching affects inference speed on GPU/MPS.

## Results

Models were trained as follows: 
- Perceptron with learning rate 0.01 and batch size 1; 
- MLP with Adam (lr=0.005), embedding dimension 256, hidden layers 128 then 64 with ReLU and dropout 0.3, training batch size 4, and a 20 epochs with early stopping.

### SST-2 (sentiment)

| System | Dev accuracy | Test accuracy |
| --- | ---: | ---: |
| Perceptron (bow + trigram + lex + neg) | 78.3% | 74% |
| MLP (256→128→64, dropout 0.3) | 77.1% | 75% |

Ablations on the development set showed that bag-of-words was the main driver for the perceptron (dropping it fell to **63.95%**). For the MLP, the mid-sized architecture with dropout worked best; removing dropout or scaling the model up or down all hurt slightly.

### Newsgroups (topic classification)

| System | Dev accuracy | Test accuracy |
| --- | ---: | ---: |
| Perceptron (bow + bigram + lex + subj) | 85.8% | 64% |
| MLP (256→128→64, dropout 0.3) | 86.0% | 67% |

On development data the two models were nearly tied, with the MLP edging ahead. Subject-line and bigram features helped the perceptron; dropout again mattered for the MLP (no dropout: **82.85%**). The larger gap between dev and test reflects distribution shift on the held-out set.

### Inference batching (MLP, Apple M4 Max / MPS)

Time to score 1,000 examples (milliseconds, averaged):

| Batch size | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Newsgroups | 165 | 87 | 45 | 41 | 30 | 29 | 30 |
| SST-2 | 171 | 91 | 44 | 31 | 28 | 21 | 23 |

Larger batches help up to roughly 16–32; past that the gains flatten.

## Setup

### Project layout

| Path | Description |
| --- | --- |
| `perceptron.py` | Feature-based multiclass perceptron: training, scoring, prediction, and CLI. |
| `multilayer_perceptron.py` | Token embedding + average-pool MLP in PyTorch, including tokenizer and DataLoader batching. |
| `features.py` | Perceptron feature extractors (bag-of-words, n-grams, subject tokens, lexicon, negation, etc.). |
| `utils.py` | Shared helpers for loading train/dev/test data, accuracy, and writing prediction CSVs. |
| `stopwords.txt` | Stopword list used by feature extractors and the MLP tokenizer. |
| `data/` | SST-2 and Newsgroups splits (`train` / `dev` / `test`). |
| `results/` | Saved model weights and prediction CSVs from trained runs. |
| `tests/` | Unit tests for features, I/O utils, the perceptron, and the MLP. |
| `pyproject.toml` | Project metadata and Python dependencies (numpy, pandas, torch, pytest, tqdm). |

### Environment

A virtual environment is recommended. With [`uv`](https://github.com/astral-sh/uv):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

### Train and predict

```sh
python perceptron.py -d newsgroups -f bow
python perceptron.py -d sst2 -f bow
python multilayer_perceptron.py -d newsgroups
```

Feature flags for the perceptron can be combined (e.g. `bow`, `bigram`, `trigram`, `subj`, `lex`, `neg`) depending on the dataset.

### Tests

From the project root:

```sh
pytest
pytest tests/test_perceptron.py
pytest tests/test_multilayer_perceptron.py
```
