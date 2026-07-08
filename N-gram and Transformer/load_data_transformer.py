import os
import random
import json

import torch
from torch.utils.data import DataLoader, Dataset

from typing import List, Any, Tuple

class TokenizationArgumentException(Exception):
    def __init__(self, message: str, error_code: int | None = None):
        super().__init__(message)
        self.error_code = error_code


def _normalize_tokenization_level(tokenization_level: str) -> str:
    level = (tokenization_level or "").strip().lower()
    if level in {"character", "char", "char_level", "character_level"}:
        return "character"
    raise TokenizationArgumentException("Transformer pipeline only supports character tokenization")


def _read_lm_lines(filepath: str) -> List[str]:
    with open(filepath, "r") as f:
        return [line.rstrip("\n") for line in f.readlines()]


def _tokenize_chars(lines: List[str]) -> List[List[str]]:
    return [list(line) for line in lines]


def _load_wer_json(filepath: str) -> List[dict]:
    with open(filepath, "r") as f:
        loaded_data = json.load(f)

    output: List[dict] = []
    for utterance_id, item in loaded_data.items():
        tokenized_sentences = [list(sentence) for sentence in item["sentences"]]
        output.append(
            {
                "id": utterance_id,
                "sentences": item["sentences"],
                "tokenized_sentences": tokenized_sentences,
                "acoustic_scores": item["acoustic_scores"],
            }
        )
    return output


def build_char_vocab(tokenized_train_data: List[List[str]]):
    chars = set()
    for sent in tokenized_train_data:
        chars.update(sent)

    itos: List[str] = ["[BOS]", "[EOS]", "[UNK]"] + sorted(chars) + ["[PAD]"]
    stoi = {tok: i for i, tok in enumerate(itos)}
    vocab = {
        "itos": itos,
        "stoi": stoi,
        "bos_id": stoi["[BOS]"],
        "eos_id": stoi["[EOS]"],
        "unk_id": stoi["[UNK]"],
        "pad_id": stoi["[PAD]"],
    }
    return vocab


class CharLMDataset(Dataset):
    def __init__(self, token_ids: List[List[int]], seq_len: int, pad_id: int):
        self.token_ids = token_ids
        self.seq_len = int(seq_len)
        self.pad_id = int(pad_id)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.token_ids[idx]
        input_ids = seq[: self.seq_len]
        target_ids = seq[1 : self.seq_len + 1]

        if len(input_ids) < self.seq_len:
            input_ids = input_ids + [self.pad_id] * (self.seq_len - len(input_ids))
        if len(target_ids) < self.seq_len:
            target_ids = target_ids + [self.pad_id] * (self.seq_len - len(target_ids))

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def load_data_transformer(
    tokenization_level: str = "character",
    *,
    seq_len: int = 128,
    batch_size: int = 64,
    seed: int = 0,
):
    """
    Function for loading data for language modeling and WER computation for the transformer models. You
    may modify the function header and outputs as necessary.

    Inputs:
        tokenization_level (str): The level at which to tokenize the input
    """
    tokenization_level = _normalize_tokenization_level(tokenization_level)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(base_dir, "data", "lm_data", "treebank-sentences-train.txt")
    dev_file = os.path.join(base_dir, "data", "lm_data", "treebank-sentences-dev.txt")
    test_file = os.path.join(base_dir, "data", "lm_data", "treebank-sentences-test.txt")
    dev_wer_file = os.path.join(base_dir, "data", "wer_data", "dev_sentences.json")
    test_wer_file = os.path.join(base_dir, "data", "wer_data", "test_sentences.json")

    train_lines = _read_lm_lines(train_file)
    dev_lines = _read_lm_lines(dev_file)
    test_lines = _read_lm_lines(test_file)

    tokenized_train = _tokenize_chars(train_lines)
    tokenized_dev = _tokenize_chars(dev_lines)
    tokenized_test = _tokenize_chars(test_lines)

    vocab = build_char_vocab(tokenized_train)
    stoi = vocab["stoi"]
    bos_id = vocab["bos_id"]
    eos_id = vocab["eos_id"]
    unk_id = vocab["unk_id"]
    pad_id = vocab["pad_id"]

    def encode(sent: List[str]) -> List[int]:
        return [bos_id] + [stoi.get(ch, unk_id) for ch in sent] + [eos_id]

    encoded_train = [encode(sent) for sent in tokenized_train]
    encoded_dev = [encode(sent) for sent in tokenized_dev]
    encoded_test = [encode(sent) for sent in tokenized_test]

    rng = random.Random(seed)
    indices = list(range(len(encoded_train)))
    rng.shuffle(indices)
    val_size = max(1, int(0.1 * len(encoded_train)))
    val_ids = set(indices[:val_size])
    train_encoded = [encoded_train[i] for i in indices if i not in val_ids]
    val_encoded = [encoded_train[i] for i in indices if i in val_ids]

    train_dataset = CharLMDataset(train_encoded, seq_len=seq_len, pad_id=pad_id)
    val_dataset = CharLMDataset(val_encoded, seq_len=seq_len, pad_id=pad_id)
    dev_dataset = CharLMDataset(encoded_dev, seq_len=seq_len, pad_id=pad_id)
    test_dataset = CharLMDataset(encoded_test, seq_len=seq_len, pad_id=pad_id)

    gen = torch.Generator()
    gen.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dev_wer_data = _load_wer_json(dev_wer_file)
    test_wer_data = _load_wer_json(test_wer_file)

    return train_loader, val_loader, dev_loader, test_loader, dev_wer_data, test_wer_data, vocab
