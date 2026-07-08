# Natural Language to SQL Translator

This project maps natural-language questions about flights to executable SQLite queries over a flight database with the help of two Transformer-based models. I compared three approaches: **fine-tuning T5-small**, **training T5-small from scratch**, and **in-context learning (ICL)** with a small Gemma LLM.

The T5 models treat NL to SQL as sequence-to-sequence generation: the encoder takes the question, and the decoder produces SQL (with light SQL canonicalization and a BOS token on targets). The LLM path prompts Gemma with the schema plus few-shot (NL, SQL) examples selected by TF-IDF similarity, then extracts the generated query. Systems are scored by **Query Exact Match** and **Record F1** (overlap of the result sets returned when predicted vs. gold SQL are run on the database).

## Results

Data: **4,225** train / **466** dev NL–SQL pairs (mean ~18 T5 tokens of NL, ~217 of SQL before preprocessing). Metrics below are Query EM and Record F1 on development unless noted; test Record F1 is from the course leaderboard.

T5 setups:
- **Fine-tuned T5:** full `google-t5/t5-small`, AdamW (lr=$3\times10^{-4}$), cosine schedule, greedy decoding (`max_new_tokens`=256), AMP; best run batch size 1, 5 epochs, patience 2 on dev Record F1.
- **T5 from scratch:** same architecture/config randomly initialized; best run batch size 8, 30 epochs, patience 5.
- **ICL (Gemma):** best prompt with hard SQL constraints + schema; $k=3$ TF-IDF-selected train examples.

### Development set

| System | Query EM | Record F1 |
| --- | ---: | ---: |
| ICL full ($k=3$, TF-IDF) | 0.0687 | 0.4140 |
| ICL $k=0$ (zero-shot) | 0.0051 | 0.1321 |
| ICL $k=1$, TF-IDF | 0.0112 | 0.2940 |
| ICL $k=3$, random examples | 0.0430 | 0.3592 |
| ICL (schema ablated) | 0.0130 | 0.1930 |
| ICL (constraints ablated) | 0.0093 | 0.1430 |
| ICL (NL/SQL tags ablated) | 0.0329 | 0.4033 |
| T5 fine-tuned (bs=1, 5 epochs) | 0.0300 | **0.6397** |
| T5 fine-tuned milestone (bs=64, 5 epochs) | 0.0150 | 0.5071 |
| T5 from scratch (bs=8, 30 epochs) | 0.0072 | 0.2085 |
| T5 from scratch milestone (bs=8, 5 epochs) | 0.0013 | 0.1180 |

Fine-tuned T5 dominates Record F1 even though ICL has higher query EM: many T5 outputs are not string-identical to gold SQL but still return overlapping result sets. For ICL, more/better examples help a lot ($k=0$ → $k=3$; TF-IDF beats random), and removing the schema or hard constraints hurts sharply.

### Test set (Record F1)

| System | Record F1 |
| --- | ---: |
| T5 fine-tuning (full model) | **0.6488** |
| T5 fine-tuning (milestone) | 0.5200 |
| T5 from scratch (full model) | 0.0744 |
| T5 from scratch (milestone) | 0.0509 |

Common failure modes across models were schema linking (wrong tables/columns for cities and airlines), dropped constraints (e.g. day-of-week or nonstop), and aggregation/grouping mistakes.

## Setup

### Project layout

| Path | Description |
| --- | --- |
| `train_t5.py` | T5 train/eval loop (fine-tune or from scratch), checkpointing, and test SQL generation. |
| `t5_utils.py` | Model init, AdamW + LR schedulers, checkpoint save/load, optional wandb. |
| `prompting.py` | Gemma ICL pipeline: prompt construction, example selection (random / TF-IDF), generation, and eval. |
| `prompting_utils.py` | Schema string formatting for prompts, SQL extraction from model text, experiment logging. |
| `load_data.py` | NL/SQL loading, T5 Datasets/DataLoaders, optional schema-conditioned prefixes for T5. |
| `utils.py` | Run SQL against `flight_database.db`, save queries/records, compute SQL EM and Record F1. |
| `evaluate.py` | CLI to score predicted `.sql` / `.pkl` records against development gold. |
| `data/` | Train/dev/test `.nl` / `.sql`, DB schema, and SQLite `flight_database.db`. |
| `results/` | Saved predicted SQL (e.g. `t5_ft_test.sql`, `t5_scr_test.sql`, `llm_test.sql`). |
| `tests/` | Unit tests for data loading and related helpers. |
| `pyproject.toml` | Project metadata and dependencies (torch, transformers, sentencepiece, pytest, etc.). |

### Environment

A virtual environment is recommended. With [`uv`](https://github.com/astral-sh/uv):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

Python **3.10** is required (`requires-python = ">=3.10,<3.11"`).

### Train and predict

Fine-tune or train T5 from scratch:

```sh
uv run train_t5.py --finetune --experiment_name ft2ep --batch_size 1 --max_n_epochs 5 --patience_epochs 2 --amp --canonicalize_sql_targets
uv run train_t5.py --experiment_name t5_scr_baseline --batch_size 8 --max_n_epochs 30 --patience_epochs 5 --amp --canonicalize_sql_targets
```

Run Gemma ICL (few-shot with TF-IDF example selection):

```sh
uv run prompting.py --shot 3 --selection tfidf --eval_split both --experiment_name icl_k3
```

### Evaluate

If you have predicted SQL and the associated database records:

```sh
uv run evaluate.py \
  --predicted_sql results/t5_ft_ft2ep_dev.sql \
  --predicted_records records/t5_ft_ft2ep_dev.pkl \
  --development_sql data/dev.sql \
  --development_records records/ground_truth_dev.pkl
```

### Tests

From the project root:

```sh
uv run pytest
uv run pytest tests/test_loaddata.py
```
