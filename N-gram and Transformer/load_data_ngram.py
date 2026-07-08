import os
import random
from collections import Counter

import torch
import json
from torch.utils.data import DataLoader, Dataset

import re
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from typing import List, Any, Tuple
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class TokenizationArgumentException(Exception):
    def __init__(self, message, error_code):
        super().__init__(message)

def custom_collate(batch):
    return batch



def train_tokenizer(files, tokenization_level):
      special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

      if tokenization_level == "word":
          tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
          tokenizer.pre_tokenizer = Whitespace()
          trainer = WordLevelTrainer(special_tokens=special_tokens)

      elif tokenization_level == "subword":
          tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
          tokenizer.pre_tokenizer = Whitespace()
          trainer = BpeTrainer(special_tokens=special_tokens)

      else:
          raise TokenizationArgumentException("Wrong argument for tokenization level")

      tokenizer.train(files, trainer=trainer)
      return tokenizer

def tokenize_txt(tokenizer, file):
    output = []
    lines = None
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        tokenized_sentence = tokenizer.encode(line)
        output.append(tokenized_sentence.tokens)
    return output

def tokenize_wer_json(tokenizer, filepath):
    with open(filepath, 'r') as f:
        loaded_data = json.load(f)

    output = []
    for utterance_id, item in loaded_data.items():
        tokenized_sentences = []
        for sentence in item["sentences"]:
            tokenized_sentences.append(tokenizer.encode(sentence).tokens)

        output.append({
            "id": utterance_id,
            "sentences": item["sentences"],
            "tokenized_sentences": tokenized_sentences,
            "acoustic_scores": item["acoustic_scores"],
        })
    return output

def tokenize_chars_txt(file):
    output = []
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        output.append(list(line.rstrip('\n')))
    return output

def tokenize_chars_wer_json(filepath):
    with open(filepath, 'r') as f:
        loaded_data = json.load(f)

    output = []
    for utterance_id, item in loaded_data.items():
        tokenized_sentences = []
        for sentence in item["sentences"]:
            tokenized_sentences.append(list(sentence))

        output.append({
            "id": utterance_id,
            "sentences": item["sentences"],
            "tokenized_sentences": tokenized_sentences,
            "acoustic_scores": item["acoustic_scores"],
        })
    return output


def _normalize_tokenization_level(tokenization_level: str) -> str:
    level = (tokenization_level or "").strip().lower()
    if level in {"character", "char", "char_level", "character_level"}:
        return "character"
    if level in {"word", "word_level"}:
        return "word"
    if level in {"subword", "subword_level", "bpe"}:
        return "subword"
    raise TokenizationArgumentException("Wrong argument for tokenization level")


def _replace_rare_tokens(tokenized_sentences: List[List[Any]], unk_token: Any = "[UNK]", min_freq: int = 3):
    counts = Counter()
    for sentence in tokenized_sentences:
        counts.update(sentence)

    def map_token(token):
        return unk_token if counts[token] < min_freq else token

    return [[map_token(token) for token in sentence] for sentence in tokenized_sentences]



def load_data_ngram(tokenization_level: str = "character"):
    """
    Function for loading data for language modeling and WER computation for the n-gram models. You
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

    if tokenization_level == "word":
        files=[train_file]
        tokenizer = train_tokenizer(files, "word")
        tokenized_train_data = tokenize_txt(tokenizer, train_file)
        tokenized_dev_data = tokenize_txt(tokenizer, dev_file)
        tokenized_test_data = tokenize_txt(tokenizer, test_file)
        tokenized_dev_wer_data = tokenize_wer_json(tokenizer, dev_wer_file)
        tokenized_test_wer_data = tokenize_wer_json(tokenizer, test_wer_file)
        
    elif tokenization_level == "subword":
        files=[train_file]
        tokenizer = train_tokenizer(files, "subword")
        tokenized_train_data = tokenize_txt(tokenizer, train_file)
        tokenized_dev_data = tokenize_txt(tokenizer, dev_file)
        tokenized_test_data = tokenize_txt(tokenizer, test_file)
        tokenized_dev_wer_data = tokenize_wer_json(tokenizer, dev_wer_file)
        tokenized_test_wer_data = tokenize_wer_json(tokenizer, test_wer_file)
        
    elif tokenization_level == "character":
        tokenized_train_data = tokenize_chars_txt(train_file)
        tokenized_dev_data = tokenize_chars_txt(dev_file)
        tokenized_test_data = tokenize_chars_txt(test_file)
        tokenized_dev_wer_data = tokenize_chars_wer_json(dev_wer_file)
        tokenized_test_wer_data = tokenize_chars_wer_json(test_wer_file)
    else:  # pragma: no cover
        raise TokenizationArgumentException("Wrong argument for tokenization level")

    if tokenization_level in {"word", "subword"}:
        tokenized_train_data = _replace_rare_tokens(tokenized_train_data, unk_token="[UNK]", min_freq=3)

    rng = random.Random(0)
    indices = list(range(len(tokenized_train_data)))
    rng.shuffle(indices)
    val_size = max(1, int(0.1 * len(tokenized_train_data)))
    val_data = [tokenized_train_data[i] for i in indices[:val_size]]
    train_data = [tokenized_train_data[i] for i in indices[val_size:]]

    return (
        train_data,
        val_data,
        tokenized_dev_data,
        tokenized_test_data,
        tokenized_dev_wer_data,
        tokenized_test_wer_data,
    )
