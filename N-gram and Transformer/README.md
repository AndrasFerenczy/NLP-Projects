# Language Modeling with N-grams and Transformers

This project compares two families of language models: count-based **n-gram LMs** and a neural **character-level Transformer LM**, on **WSJ** sentence perplexity (Wall Street Journal / Penn Treebank) and **HUB** ASR *n*-best list reranking (word error rate).

The n-gram models are word-, BPE subword-, and character-level LMs with maximum-likelihood estimates, Laplace smoothing, and linear interpolation. They count contexts from tokenized WSJ training text and score sequences with optional smoothing so rare or unseen contexts do not collapse to zero probability. The Transformer is a decoder-only causal LM over characters: it embeds tokens with sinusoidal positional encodings, stacks self-attention blocks, and is trained with AdamW and next-token cross-entropy (padding ignored). For HUB, each candidate transcription is scored as acoustic score + LM log-probability, and the best candidate is selected per spoken sentence.

## Results

Models were trained on WSJ train (90% fit / 10% validation after a deterministic shuffle). Perplexity is base-2 on WSJ; WER is from HUB *n*-best reranking. Test WER comes from the course leaderboard (hidden labels).

### N-gram models

| System | Dev PPL (WSJ) | Test PPL (WSJ) | Dev WER (HUB) | Test WER (HUB) |
| --- | ---: | ---: | ---: | ---: |
| Word-1 (MLE) | 900.46 | — | 9.58% | — |
| Word-2 (MLE) | ∞ | — | 10.47% | — |
| Word-2-Interp | 183.82 | 178.21 | 7.80% | — |
| Word-3-Interp | 163.87 | — | 8.46% | — |
| BPE-3-Interp | 229.09 | — | 8.24% | — |
| BPE-3-Interp+Lap ($k=10^{-4}$) | 208.49 | — | 8.46% | 10.4% |
| Char-3-Interp+Lap | 7.87 | — | 8.69% | — |
| Char-5-Interp+Lap | 3.90 | 3.86 | 7.57% | 11.39% |

Unsmoothed higher-order n-grams hit zero-probability events (infinite perplexity). Linear interpolation fixes most of that; for characters, longer context ($n=5$) plus light Laplace gave the best n-gram HUB development WER (**7.57%**). Word-level interpolation was strong on HUB as well (Word-2-Interp: **7.80%**), while character models dominate WSJ perplexity because the vocabulary is tiny.

### Transformer models (character-level)

| System | Layers / hidden / heads | Dev PPL (WSJ) | Test PPL (WSJ) | Dev WER (HUB) | Test WER (HUB) |
| --- | --- | ---: | ---: | ---: | ---: |
| TF-Small-E1 | 2 / 128 / 4, 1 epoch | 11.03 | — | 9.58% | — |
| TF-Small-E3 | 2 / 128 / 4, 3 epochs | 10.40 | — | 9.58% | — |
| TF-Small-L4 | 4 / 128 / 4 | 7.60 | — | 8.46% | — |
| TF-Med-L4H8 | 4 / 256 / 8 | 5.51 | — | 9.35% | — |
| TF-Med-L4H4 | 4 / 128 / 4, lr=$10^{-3}$ | 9.91 | — | 9.80% | — |
| TF-Large | 6 / 512 / 8, 5 epochs | 3.38 | 3.34 | 6.46% | 11% |

Larger models and more layers cut WSJ perplexity sharply. **TF-Large** was best on development HUB WER (**6.46%**) and also best on WSJ test perplexity (**3.34**). Test WER rose relative to dev for both the best n-gram and the Transformer, reflecting shift on the held-out HUB set.

## Setup

### Project layout

| Path | Description |
| --- | --- |
| `n_gram.py` | Count-based n-gram LM: MLE, Laplace, linear interpolation, training CLI, generation, and HUB reranking. |
| `transformer.py` | Character-level decoder-only Transformer in PyTorch: train, checkpoint, perplexity, generation, and HUB reranking. |
| `load_data_ngram.py` | WSJ/HUB loading and tokenization for n-grams (word, BPE subword, character) plus rare-token → `[UNK]` handling. |
| `load_data_transformer.py` | Character vocab, BOS/EOS/PAD encoding, train/val split, and DataLoaders for the Transformer. |
| `perplexity.py` | Base-2 corpus perplexity for list-based (n-gram) and batched (Transformer) data. |
| `wer.py` | Acoustic + LM score reranking of HUB *n*-best lists and local development WER. |
| `evaluation.py` | Convenience CLI to compute WER from a predictions CSV against `dev_ground_truths.csv`. |
| `data/lm_data/` | WSJ train / dev / test sentence files for language modeling. |
| `data/wer_data/` | HUB dev/test candidate JSONs and development ground-truth CSV. |
| `results/` | Final test prediction CSVs for leaderboard-style submission. |
| `generations/` | Saved generation example dumps from n-gram and Transformer runs. |
| `tests/` | Unit tests for n-gram and Transformer modules. |
| `pyproject.toml` | Project metadata and Python dependencies (torch, tokenizers, evaluate, pytest, etc.). |

### Environment

A virtual environment is recommended. With [`uv`](https://github.com/astral-sh/uv):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

### Train and predict

```sh
uv run n_gram.py --tokenization_level word --n=3 --experiment_name=trigram --num_samples=100
uv run n_gram.py --tokenization_level character --n=5 --laplace_k=0.0001 --experiment_name=char5
uv run transformer.py --num_layers=4 --hidden_dim=256 --experiment_name=transformer
```

Useful n-gram flags: `--tokenization_level {word,subword,character}`, `--no_interpolate`, `--laplace_k`, `--interp_lambdas`. Useful Transformer flags: `--num_layers`, `--hidden_dim`, `--num_heads`, `--ff_dim`, `--dropout_p`, `--learning_rate`, `--max_epochs`, `--load_checkpoint`.

### Tests

From the project root:

```sh
uv run pytest
uv run pytest tests/test_n_gram.py
uv run pytest tests/test_transformer.py
```
