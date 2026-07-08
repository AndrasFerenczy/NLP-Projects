import math
import os
import random
from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import nltk
from torch.utils.data import DataLoader, Dataset
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# --- Paths and Constants Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "training")

PATH_CONLL = os.path.join(TRAIN_DATA_DIR, "training-data.1m.conll")
PATH_SHARDS = os.path.join(TRAIN_DATA_DIR, "training-monolingual.tokenized.shuffled")
PATH_EVAL = os.path.join(DATA_DIR, "isolated_similarity")
PATH_CACHE = os.path.join(DATA_DIR, "cache")

TOKEN_PAD = "<PAD>"
TOKEN_UNK = "<UNK>"

# N-gram settings
MIN_NGRAM_SIZE = 3
MAX_NGRAM_SIZE = 6
SUBWORD_PAD_LEN = 50

# Shard processing defaults
OVERLAPPING_SHARDS = 4
DEFAULT_EXTRA_SHARDS = 5


def extract_subwords(word: str) -> List[str]:
    """Generates the full token in brackets and all distinct character n-grams."""
    enclosed = f"<{word}>"
    ngrams = [enclosed]
    chars_len = len(enclosed)
    
    for n in range(MIN_NGRAM_SIZE, MAX_NGRAM_SIZE + 1):
        for start_idx in range(chars_len - n + 1):
            ngrams.append(enclosed[start_idx : start_idx + n])
            
    return ngrams


def pad_truncate_sequence(token_ids: List[int], target_len: int) -> List[int]:
    """Ensures a sequence is exactly `target_len` integers long."""
    current_len = len(token_ids)
    if current_len >= target_len:
        return token_ids[:target_len]
    return token_ids + [0] * (target_len - current_len)


def parse_conll_corpus(file_path: str):
    """
    Reads a CONLL-U style formatted file and returns a list of parsed sentences.
    Skips over comments and multi-word token designations.
    """
    sentences = []
    current_sentence = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 8:
                continue

            # Ignore entries like 1-2
            if not parts[0].isdigit():
                continue

            token_data = {
                "idx": int(parts[0]),
                "text": parts[1].lower(),
                "head_idx": int(parts[6]) if parts[6].isdigit() else 0,
                "relation": parts[7],
            }
            current_sentence.append(token_data)

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


class SkipGramPairsDataset(Dataset):
    """PyTorch Dataset yielding target n-grams and center context IDs."""

    def __init__(self, target_indices: np.ndarray, context_indices: np.ndarray,
                 ngram_matrix: torch.Tensor, total_targets: int, total_contexts: int):
        self.target_indices = target_indices
        self.context_indices = context_indices
        self.ngram_matrix = ngram_matrix
        self.vocab_size = total_targets
        self.context_vocab_size = total_contexts

        self.neg_weights = None
        self.token_to_id = None

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, index):
        t_idx = self.target_indices[index]
        c_idx = self.context_indices[index]
        return self.ngram_matrix[t_idx], torch.tensor(c_idx, dtype=torch.long)


class WordSimilarityEvalSet:
    """Holds encoded subword tensors for two corresponding word lists."""
    def __init__(self, word1_tensor: torch.LongTensor, word2_tensor: torch.LongTensor):
        self.word1_ids = word1_tensor
        self.word2_ids = word2_tensor


def compute_subsampling_probabilities(freq_dist: Counter, total_words: int, t: float = 1e-4) -> dict:
    """
    Subsample words using the official Word2Vec paper formula:
       P(keep_i) = sqrt(t / f(w_i))
    """
    import math
    
    keep_probs = {}
    for word, count in freq_dist.items():
        freq_ratio = count / total_words
        
        # Pure abstract paper formula: sqrt(t/f)
        paper_prob = math.sqrt(t / freq_ratio)
        
        # Cap probability at 1.0
        keep_probs[word] = min(1.0, paper_prob)
            
    return keep_probs


def get_cache_filename(freq_cutoff: int, num_shards: int) -> str:
    return f"processed_data_min{freq_cutoff}_sh{num_shards}.pt"


def dump_to_cache(file_path: str, targets, contexts, ngrams, neg_dist, vocab_map, num_ctx):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    payload = {
        "targets": targets,
        "contexts": contexts,
        "ngrams": ngrams,
        "neg_dist": neg_dist,
        "vocab_map": vocab_map,
        "num_ctx": num_ctx,
    }
    torch.save(payload, file_path)
    print(f"[CACHE] Saved preprocessed data to {file_path}")


def restore_from_cache(file_path: str):
    print(f"[CACHE] Restoring from {file_path} ...")
    content = torch.load(file_path, weights_only=False)
    print(f"[CACHE] Loaded {len(content['targets']):,} training pairs.")
    return content


def construct_training_pipeline(min_count: int, num_additional_shards: int):
    """
    End-to-end data processing function: parses text files, generates vocabularies,
    and extracts context window/dependency pairs.
    """
    print("[INIT] Parsing CONLL corpus for dependencies...")
    conll_sentences = parse_conll_corpus(PATH_CONLL)
    print(f"[INIT] Parsed {len(conll_sentences):,} sentences from CONLL.")

    shard_names = sorted(
        name for name in os.listdir(PATH_SHARDS)
        if name.startswith("news.en-") and name.endswith("-of-00100")
    )
    
    selected_shards = shard_names[OVERLAPPING_SHARDS : OVERLAPPING_SHARDS + num_additional_shards]
    shard_full_paths = [os.path.join(PATH_SHARDS, name) for name in selected_shards]

    text_shard_sentences: List[List[str]] = []
    for idx, path in enumerate(shard_full_paths):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = word_tokenize(line.strip().lower())
                if len(tokens) >= 2:
                    text_shard_sentences.append(tokens)
        print(f"[INIT] Finished shard {idx+1}/{len(shard_full_paths)}. Accumulated {len(text_shard_sentences):,} raw text sentences.")

    print("\n[VOCAB] Constructing subword mappings...")
    subword_counts = Counter()
    word_counts = Counter()
    total_token_count = 0

    for sentence in conll_sentences:
        for node in sentence:
            total_token_count += 1
            word = node["text"]
            word_counts[word] += 1
            subword_counts.update(extract_subwords(word))

    for sentence in text_shard_sentences:
        for token in sentence:
            total_token_count += 1
            word_counts[token] += 1
            subword_counts.update(extract_subwords(token))

    vocab_to_idx = {TOKEN_PAD: 0, TOKEN_UNK: 1}
    for item, qty in sorted(subword_counts.items(), key=lambda kv: -kv[1]):
        if qty >= min_count:
            vocab_to_idx[item] = len(vocab_to_idx)

    ctx_to_idx = {TOKEN_PAD: 0, TOKEN_UNK: 1}

    print(f"[VOCAB] Found {len(vocab_to_idx):,} unique subwords and {len(word_counts):,} unique words.")

    discard_probs = compute_subsampling_probabilities(word_counts, total_token_count)

    tracker = {}
    ngram_mapping: List[List[int]] = []

    def lookup_or_create(word_str: str) -> int:
        if word_str in tracker:
            return tracker[word_str]
            
        valid_ids = [vocab_to_idx[s] for s in extract_subwords(word_str) if s in vocab_to_idx]
        if not valid_ids:
            tracker[word_str] = -1
            return -1
            
        idx = len(ngram_mapping)
        tracker[word_str] = idx
        ngram_mapping.append(pad_truncate_sequence(valid_ids, SUBWORD_PAD_LEN))
        return idx

    target_items, context_items = [], []

    print("\n[PAIRS] Extracting pairs using dependency trees...")
    for s_idx, sentence in enumerate(conll_sentences):
        node_map = {node["idx"]: node for node in sentence}

        for node in sentence:
            w = node["text"]
            if random.random() > discard_probs.get(w, 1.0):
                continue

            matrix_row = lookup_or_create(w)
            if matrix_row < 0:
                continue

            contexts_for_word = []
            parent_id = node["head_idx"]
            
            if parent_id != 0 and parent_id in node_map:
                parent = node_map[parent_id]
                contexts_for_word.append(f"{parent['text']}/{node['relation']}^")

            for peer in sentence:
                if peer["head_idx"] == node["idx"]:
                    contexts_for_word.append(f"{peer['text']}/{peer['relation']}")

            for ctx_str in contexts_for_word:
                root = ctx_str.split("/")[0]
                if word_counts.get(root, 0) < min_count:
                    continue
                if ctx_str not in ctx_to_idx:
                    ctx_to_idx[ctx_str] = len(ctx_to_idx)
                
                target_items.append(matrix_row)
                context_items.append(ctx_to_idx[ctx_str])

    dep_pair_count = len(target_items)
    print(f"[PAIRS] Captured {dep_pair_count:,} dependency relations.")
    
    # Free memory
    del conll_sentences

    WINDOW = 2
    print(f"\n[PAIRS] Extracting linear context pairs (window size = {WINDOW})...")

    for s_idx, sentence in enumerate(text_shard_sentences):
        filtered_sentence = [token for token in sentence if random.random() <= discard_probs.get(token, 1.0)]

        for i, center in enumerate(filtered_sentence):
            matrix_row = lookup_or_create(center)
            if matrix_row < 0:
                continue

            start_bound = max(0, i - WINDOW)
            end_bound = min(len(filtered_sentence), i + WINDOW + 1)
            
            for j in range(start_bound, end_bound):
                if i == j:
                    continue
                    
                context_word = filtered_sentence[j]
                if word_counts.get(context_word, 0) < min_count:
                    continue
                    
                if context_word not in ctx_to_idx:
                    ctx_to_idx[context_word] = len(ctx_to_idx)
                    
                target_items.append(matrix_row)
                context_items.append(ctx_to_idx[context_word])

    linear_pair_count = len(target_items) - dep_pair_count
    print(f"[PAIRS] Captured {linear_pair_count:,} linear relations.")
    print(f"[PAIRS] Total contexts tracked: {len(ctx_to_idx):,}")

    del text_shard_sentences

    # Numpy arrays & memory cleanup
    np_targets = np.array(target_items, dtype=np.int32)
    np_contexts = np.array(context_items, dtype=np.int32)
    torch_ngrams = torch.tensor(ngram_mapping, dtype=torch.long)
    
    del target_items, context_items, ngram_mapping

    # Negative sampling weights calculation
    bincounts = Counter(np_contexts.tolist())
    dist = torch.zeros(len(ctx_to_idx))
    for c_id, freq in bincounts.items():
        dist[c_id] = freq
        
    dist[0] = 0.0
    dist[1] = 0.0
    dist = dist ** 0.75

    return {
        "targets": np_targets,
        "contexts": np_contexts,
        "ngrams": torch_ngrams,
        "neg_dist": dist,
        "vocab_map": vocab_to_idx,
        "num_ctx": len(ctx_to_idx),
    }


def load_data_word2vec(use_additional_data: Optional[bool] = False,
                       min_freq: int = 5,
                       window_size: int = 5,
                       batch_size: int = 16384,
                       num_extra_shards: int = DEFAULT_EXTRA_SHARDS):
    """
    Main entrypoint to load data for Word2Vec training.
    Caches intermediate steps to save redundant computation time.
    """
    cache_path = os.path.join(PATH_CACHE, get_cache_filename(min_freq, num_extra_shards))
    
    if os.path.isfile(cache_path):
        payload = restore_from_cache(cache_path)
    else:
        payload = construct_training_pipeline(min_freq, num_extra_shards)
        dump_to_cache(cache_path, **payload)

    dataset_obj = SkipGramPairsDataset(
        target_indices=payload["targets"], 
        context_indices=payload["contexts"], 
        ngram_matrix=payload["ngrams"],
        total_targets=len(payload["vocab_map"]), 
        total_contexts=payload["num_ctx"]
    )
    dataset_obj.neg_weights = payload["neg_dist"]
    dataset_obj.token_to_id = payload["vocab_map"]

    loader_obj = DataLoader(
        dataset_obj, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=4,
        persistent_workers=True
    )

    def prepare_eval_file(path: str) -> WordSimilarityEvalSet:
        data_df = pd.read_csv(path)
        w1_encodings, w2_encodings = [], []
        
        for idx in range(len(data_df)):
            entry = data_df.iloc[idx]
            w1 = entry["word1"].lower()
            w2 = entry["word2"].lower()
            
            for word_obj, target_list in [(w1, w1_encodings), (w2, w2_encodings)]:
                subword_tokens = extract_subwords(word_obj)
                token_ids = [payload["vocab_map"].get(t, 1) for t in subword_tokens]
                target_list.append(pad_truncate_sequence(token_ids, SUBWORD_PAD_LEN))
                
        return WordSimilarityEvalSet(
            torch.tensor(w1_encodings, dtype=torch.long),
            torch.tensor(w2_encodings, dtype=torch.long)
        )

    dev_set = prepare_eval_file(os.path.join(PATH_EVAL, "isolated_dev_x.csv"))
    test_set = prepare_eval_file(os.path.join(PATH_EVAL, "isolated_test_x.csv"))

    print("\n[READY] Dataloaders and evaluation assets constructed successfully.\n")
    return dataset_obj, loader_obj, dev_set, test_set
