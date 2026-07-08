import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle
import json
from functools import lru_cache

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

def _basic_tokens(text: str) -> set[str]:
    # Conservative tokenization for overlap scoring.
    tokens = set()
    for chunk in re.split(r"[^A-Za-z0-9_]+", text.lower()):
        if not chunk:
            continue
        tokens.add(chunk)
        if "_" in chunk:
            tokens.update([p for p in chunk.split("_") if p])
    return tokens

@lru_cache(maxsize=4)
def _load_schema(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def _build_schema_index(schema_json: dict) -> dict:
    """
    Build a lightweight index for overlap-based schema conditioning.

    Returns:
      {
        "tables": {
          table: {
            "table_tokens": set[str],
            "columns": {col: {"col_tokens": set[str], "utt": str}},
          }
        }
      }
    """
    ents = schema_json.get("ents", {})
    tables = {}
    token_table_df: Counter[str] = Counter()
    token_col_df: Counter[str] = Counter()
    for table, cols in ents.items():
        col_index = {}
        table_tokens = _basic_tokens(table)
        for col, meta in cols.items():
            utt = str(meta.get("utt", ""))
            col_tokens = _basic_tokens(col) | _basic_tokens(utt)
            col_index[col] = {"col_tokens": col_tokens, "utt": utt}
            table_tokens |= col_tokens
            token_col_df.update(set(col_tokens))
        tables[table] = {"table_tokens": table_tokens, "columns": col_index}
        token_table_df.update(set(table_tokens))
    return {"tables": tables, "token_table_df": token_table_df, "token_col_df": token_col_df}

@lru_cache(maxsize=4)
def _get_schema_index(schema_path: str) -> dict:
    return _build_schema_index(_load_schema(schema_path))

def build_schema_prefix(
    nl: str,
    schema_path: str,
    max_tables: int = 5,
    max_columns_per_table: int = 8,
    include_utts: bool = False,
) -> str:
    """
    Produce a compact schema prefix filtered by token overlap with the NL query.
    """
    idx = _get_schema_index(schema_path)
    nl_tokens = _basic_tokens(nl)

    def w_table(tok: str) -> float:
        df = idx["token_table_df"].get(tok, 1)
        return 1.0 / float(df)

    def w_col(tok: str) -> float:
        df = idx["token_col_df"].get(tok, 1)
        return 1.0 / float(df)

    table_scores = []
    for table, info in idx["tables"].items():
        overlap_toks = nl_tokens & info["table_tokens"]
        if overlap_toks:
            score = sum(w_table(t) for t in overlap_toks)
            table_scores.append((score, table))
    table_scores.sort(reverse=True)

    # If overlap is empty, fall back to a small, fixed set of common tables
    # rather than dumping the entire schema.
    if not table_scores:
        fallback = ["flight", "airport_service", "city", "airport", "airline"]
        selected_tables = [t for t in fallback if t in idx["tables"]][:max_tables]
    else:
        selected_tables = [t for _, t in table_scores[:max_tables]]

    # Heuristic: many flight queries mention cities/airports without saying "city"/"airport".
    # If flight is relevant, force-include its common join tables, prioritizing them over
    # less-informative matches like "time_zone" that happen to overlap on tokens like "from".
    if "flight" in selected_tables:
        related = [t for t in ["airport_service", "city", "airport", "airline"] if t in idx["tables"]]
        prioritized = ["flight"] + [t for t in related if t != "flight"]
        for t in selected_tables:
            if t not in prioritized:
                prioritized.append(t)
        selected_tables = prioritized[:max_tables]

    parts = []
    for table in selected_tables:
        cols = idx["tables"][table]["columns"]
        col_scores = []
        for col, meta in cols.items():
            overlap_toks = nl_tokens & meta["col_tokens"]
            if overlap_toks:
                score = sum(w_col(t) for t in overlap_toks)
                col_scores.append((score, col))
        col_scores.sort(reverse=True)

        if col_scores:
            chosen_cols = [c for _, c in col_scores[:max_columns_per_table]]
        else:
            # If nothing overlaps, keep a small deterministic slice of columns.
            chosen_cols = list(cols.keys())[: min(max_columns_per_table, len(cols))]

        if include_utts:
            rendered_cols = [f"{c}({cols[c]['utt']})" if cols[c]["utt"] else c for c in chosen_cols]
        else:
            rendered_cols = chosen_cols
        parts.append(f"{table}({', '.join(rendered_cols)})")

    return "schema: " + " ; ".join(parts)

def canonicalize_sql(sql: str) -> str:
    """
    Lightweight SQL canonicalization to reduce target-side formatting variance.

    Notes:
      - This is intentionally conservative: it should not change query semantics.
      - It is opt-in (see `T5Dataset(..., canonicalize_sql=True)`), to avoid changing
        the starter behavior and unit-test fixtures.
    """
    s = sql.strip()
    # Strip trailing semicolons (some lines have them; most do not).
    s = re.sub(r";+\s*$", "", s)
    # Fix common tokenization-unfriendly patterns in this dataset (e.g., "AND(").
    s = re.sub(r"\b(AND|OR|NOT)\(", r"\1 (", s)
    # Collapse any accidental repeated whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s

class T5Dataset(Dataset):

    def __init__(
        self,
        data_folder,
        split,
        canonicalize_sql_targets: bool = False,
        schema_conditioning: bool = False,
        schema_path: str | None = None,
        schema_max_tables: int = 5,
        schema_max_columns_per_table: int = 8,
        schema_include_utts: bool = False,
        schema_separator: str = " | ",
    ):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose (e.g., "<extra_id_0>").
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.decoder_bos_token = "<extra_id_0>"
        self.decoder_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.decoder_bos_token)
        self.canonicalize_sql_targets = canonicalize_sql_targets
        self.schema_conditioning = schema_conditioning
        self.schema_path = schema_path or os.path.join(data_folder, "flight_database.schema")
        self.schema_max_tables = schema_max_tables
        self.schema_max_columns_per_table = schema_max_columns_per_table
        self.schema_include_utts = schema_include_utts
        self.schema_separator = schema_separator

        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_lines = load_lines(nl_path)

        sql_lines = None
        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            sql_lines = load_lines(sql_path)
            assert len(nl_lines) == len(sql_lines)

        data = []
        for i, nl in enumerate(nl_lines):
            encoder_text = nl
            if self.schema_conditioning:
                prefix = build_schema_prefix(
                    nl,
                    schema_path=self.schema_path,
                    max_tables=self.schema_max_tables,
                    max_columns_per_table=self.schema_max_columns_per_table,
                    include_utts=self.schema_include_utts,
                )
                encoder_text = prefix + self.schema_separator + "question: " + nl

            enc = tokenizer(
                encoder_text,
                add_special_tokens=True,
                return_attention_mask=True,
                padding=False,
                truncation=False,
            )
            encoder_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
            encoder_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)

            if split == "test":
                sql_line = self.decoder_bos_token
            else:
                target_sql = sql_lines[i]
                if self.canonicalize_sql_targets:
                    target_sql = canonicalize_sql(target_sql)
                sql_line = f"{self.decoder_bos_token}{target_sql}"

            dec = tokenizer(
                sql_line,
                add_special_tokens=True,
                return_attention_mask=False,
                padding=False,
                truncation=False,
            )
            decoder_ids = torch.tensor(dec["input_ids"], dtype=torch.long)

            data.append(
                {
                    "encoder_ids": encoder_ids,
                    "encoder_mask": encoder_mask,
                    "decoder_ids": decoder_ids,
                    "sql_line": sql_line,
                }
            )

        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data[idx]

        #sql_line is just a string of the sql command with the special bos token added in the beginning. 
        return data_dict["encoder_ids"], data_dict["encoder_mask"], data_dict["decoder_ids"], \
            data_dict["sql_line"]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to the decoder (only to be used in evaluation)
    '''
    encoder_ids, encoder_masks, decoder_ids, _ = zip(*batch)

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_masks = pad_sequence(encoder_masks, batch_first=True, padding_value=0)

    decoder_ids = pad_sequence(decoder_ids, batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = decoder_ids[:, :-1]
    decoder_targets = decoder_ids[:, 1:]
    initial_decoder_inputs = decoder_inputs[:, :1]

    return encoder_ids, encoder_masks, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to the decoder (only to be used in evaluation)
    '''
    encoder_ids, encoder_masks, decoder_ids, _ = zip(*batch)

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_masks = pad_sequence(encoder_masks, batch_first=True, padding_value=0)

    decoder_ids = pad_sequence(decoder_ids, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = decoder_ids[:, :1]

    return encoder_ids, encoder_masks, initial_decoder_inputs
    

def get_dataloader(
    batch_size,
    split,
    canonicalize_sql_targets: bool = False,
    schema_conditioning: bool = False,
    schema_path: str | None = None,
    schema_max_tables: int = 5,
    schema_max_columns_per_table: int = 8,
    schema_include_utts: bool = False,
    schema_separator: str = " | ",
):
    data_folder = 'data'
    dset = T5Dataset(
        data_folder,
        split,
        canonicalize_sql_targets=canonicalize_sql_targets,
        schema_conditioning=schema_conditioning,
        schema_path=schema_path,
        schema_max_tables=schema_max_tables,
        schema_max_columns_per_table=schema_max_columns_per_table,
        schema_include_utts=schema_include_utts,
        schema_separator=schema_separator,
    )
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(
    batch_size,
    test_batch_size,
    canonicalize_sql_targets: bool = False,
    schema_conditioning: bool = False,
    schema_path: str | None = None,
    schema_max_tables: int = 5,
    schema_max_columns_per_table: int = 8,
    schema_include_utts: bool = False,
    schema_separator: str = " | ",
):
    train_loader = get_dataloader(
        batch_size,
        "train",
        canonicalize_sql_targets=canonicalize_sql_targets,
        schema_conditioning=schema_conditioning,
        schema_path=schema_path,
        schema_max_tables=schema_max_tables,
        schema_max_columns_per_table=schema_max_columns_per_table,
        schema_include_utts=schema_include_utts,
        schema_separator=schema_separator,
    )
    dev_loader = get_dataloader(
        test_batch_size,
        "dev",
        canonicalize_sql_targets=canonicalize_sql_targets,
        schema_conditioning=schema_conditioning,
        schema_path=schema_path,
        schema_max_tables=schema_max_tables,
        schema_max_columns_per_table=schema_max_columns_per_table,
        schema_include_utts=schema_include_utts,
        schema_separator=schema_separator,
    )
    test_loader = get_dataloader(
        test_batch_size,
        "test",
        canonicalize_sql_targets=canonicalize_sql_targets,
        schema_conditioning=schema_conditioning,
        schema_path=schema_path,
        schema_max_tables=schema_max_tables,
        schema_max_columns_per_table=schema_max_columns_per_table,
        schema_include_utts=schema_include_utts,
        schema_separator=schema_separator,
    )
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    """
    Load raw (non-tokenized) data splits for Part 3 prompting / ICL experiments.

    Returns:
        train_x: List[str] from {data_folder}/train.nl
        train_y: List[str] from {data_folder}/train.sql
        dev_x:   List[str] from {data_folder}/dev.nl
        dev_y:   List[str] from {data_folder}/dev.sql
        test_x:  List[str] from {data_folder}/test.nl
    """
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))

    if len(train_x) != len(train_y):
        raise ValueError(f"train.nl ({len(train_x)}) and train.sql ({len(train_y)}) length mismatch")
    if len(dev_x) != len(dev_y):
        raise ValueError(f"dev.nl ({len(dev_x)}) and dev.sql ({len(dev_y)}) length mismatch")

    return train_x, train_y, dev_x, dev_y, test_x
