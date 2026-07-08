# Word Embeddings with Word2Vec

This project implements a **skip-gram Word2Vec** model with negative sampling from scratch in PyTorch, and evaluates the learned embeddings on an **isolated word-pair similarity** task (Spearman correlation against human ratings). 

The scoring system scored **0.395** on the hidden test set ([course leaderboard](https://github.com/Cornell-Tech-CS5744-Spring-2026/leaderboards/blob/main/a2/leaderboard.csv)). My implementation achieved a score which is second best among all the previous and current year's students.

The model learns target and context embedding matrices from large news corpora. Contexts come from dependency parses (on a 1M-sentence CoNLL set) and from sliding windows (on additional shards of the 1B-word dataset). Training uses frequency-smoothed negative sampling, word and character-ngram (subword) vocabularies for better OOV handling, and optional WordNet retrofitting to pull synonyms closer after training. The goal was to build competitive embeddings for similarity prediction and to measure how data scale, preprocessing, and post-processing affect Spearman correlation.

## Results

Models were trained as follows:
- Skip-gram Word2Vec with embedding dim 200, 10 negative samples (unigram raised to 0.75), SGD with lr scheduling from 0.0001, batch size 16384 sentences, and 3 epochs on the final run;
- Training data: all 1M dependency-parsed sentences plus 5 shards from the 1B-word corpus (~1.5M sentences total);
- Subword vocab (~500K character n-grams) alongside a ~70K word vocab; frequent-word downsampling during training;
- Post-training WordNet retrofitting (embeddings left unnormalized).

### Isolated word-pair similarity (Spearman correlation, %)

| System | Dev | Test |
| --- | ---: | ---: |
| word2vec (ep: 3, 1M + 5 shards) | 51.02 | 39.53 |
| word2vec (ep: 1, 1M) | 45.19 | 35.55 |
| word2vec (ep: 1, 1M, w/o retrofitting) | 30.03 | — |
| word2vec (ep: 1, 1M, w/o retrofitting & freq. word elimination) | 24.15 | — |
| word2vec (milestone) | — | 16.52 |

The largest gains on development data came from WordNet retrofitting (+15 points vs the same 1-epoch model without it) and from more data plus more epochs (51.02% final vs 45.19% at epoch 1 on 1M only). Frequent-word elimination also helped meaningfully. Dev correlation rose over the three training epochs (one epoch = one full pass over the 1.5M-sentence training set). The gap between final dev (51.02%) and test (39.53%) reflects distribution shift on the held-out pairs.

## Setup

### Project layout

| Path | Description |
| --- | --- |
| `word2vec.py` | Skip-gram Word2Vec model, negative-sampling loss, training loop, WordNet retrofit, and CLI. |
| `load_data.py` | Corpus loading, tokenization, vocab/subword construction, context windows (dependency + linear), and DataLoaders. |
| `similarity.py` | Scores word-pair cosine/dot similarities from two embedding dump files; writes a prediction CSV to stdout. |
| `evaluate.py` | Spearman correlation between predicted similarity scores and gold development labels. |
| `utils.py` | Helpers for similarity scores and Spearman correlation used during training eval. |
| `data/` | Training corpora (`training/`), isolated similarity splits (`isolated_similarity/`), and optional caches. |
| `results/` | Saved model weights and submitted test embeddings (`word2vec_isol_test_words{1,2}_embeddings.txt`). |
| `tests/` | Unit tests for the Word2Vec model and loss. |
| `pyproject.toml` | Project metadata and Python dependencies (numpy, pandas, torch, nltk, scipy, conllu, pytest, tqdm). |

### Environment

A virtual environment is recommended. With [`uv`](https://github.com/astral-sh/uv):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

### Train and evaluate

```sh
python word2vec.py
```

To score word pairs from embedding dumps and evaluate against development labels:

```sh
python similarity.py \
  --embedding1 results/word2vec_isol_test_words1_embeddings.txt \
  --embedding2 results/word2vec_isol_test_words2_embeddings.txt \
  --words data/isolated_similarity/isolated_dev_x.csv \
  > prediction.csv

python evaluate.py \
  --predicted prediction.csv \
  --development data/isolated_similarity/isolated_dev_y.csv
```

### Tests

From the project root:

```sh
pytest
pytest tests/test_word2vec.py
```
