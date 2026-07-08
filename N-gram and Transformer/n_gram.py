"""
n-gram language model for Assignment 2: Starter code.
"""

import os
import sys
import argparse
from typing import Dict, List, Any
from tqdm import tqdm
from collections import Counter
import math
import random
import bisect
import numpy as np

from load_data_ngram import load_data_ngram as load_data
from perplexity import evaluate_perplexity
from wer import rerank_sentences_for_wer, compute_wer

def get_args():
    """
    You may freely add new command line arguments to this function.
    """
    parser = argparse.ArgumentParser(description='n-gram model')
    parser.add_argument('-t', '--tokenization_level', type=str, default='character',
                        help="At what level to tokenize the input data")
    parser.add_argument('-n', '--n', type=int, default=1,
                        help="The value of n to use for the n-gram model")
    parser.add_argument('--laplace_k', type=float, default=1e-4,
                        help="Add-k (Laplace) smoothing strength; 0.0 disables smoothing")
    parser.add_argument('--no_interpolate', action='store_true',
                        help="Disable linear interpolation smoothing (enabled by default)")
    parser.add_argument('--interp_lambdas', type=str, default="auto",
                        help="Comma-separated interpolation weights (low->high order), or 'auto'")

    parser.add_argument('-e', '--experiment_name', type=str, default='testing',
                        help="What should we name our experiment?")
    parser.add_argument('-s', '--num_samples', type=int, default=10,
                        help="How many samples should we get from our model??")
    parser.add_argument('-x', '--max_steps', type=int, default=40,
                        help="What should the maximum output length of our samples be?")

    args = parser.parse_args()
    return args

class NGramLM():
    """
    N-gram language model
    """

    def __init__(self, n: int, laplace_k: float = 0.0, interpolation_lambdas: List[float] | None = None):
        """
        Initializes the n-gram model. You may add keyword arguments to this function
        to modify the behavior of the n-gram model. The default behavior for unit tests should
        be that of an n-gram model without any label smoothing.

        Important for unit tests: If you add <bos> or <eos> tokens to model inputs, this should 
        be done in data processing, outside of the NGramLM class. 

        Inputs:
            n (int): The value of n to use in the n-gram model
        """
        self.n = n
        self.unk_token = "[UNK]"
        self.laplace_k = float(laplace_k)
        self.interpolation_lambdas = interpolation_lambdas

    def log_probability(self, model_input: List[Any], base=np.e):
        """
        Returns the log-probability of the provided model input.

        Inputs:
            model_input (List[Any]): The list of tokens associated with the input text.
            base (float): The base with which to compute the log-probability
        """
        n = self.n
        if not model_input:
            return 0.0

        tokens = model_input
        vocab = getattr(self, "vocab", None)
        unk_token = getattr(self, "unk_token", None)
        if vocab is not None and unk_token is not None:
            tokens = [token if token in vocab else unk_token for token in tokens]

        if base == 2:
            log_fn = math.log2
            inv_log_base = None
        else:
            log_fn = None
            inv_log_base = 1.0 / math.log(base)

        if getattr(self, "interpolation_lambdas", None) is not None and n >= 2:
            return self._log_probability_interpolated(tokens, log_fn, inv_log_base)

        k = getattr(self, "laplace_k", 0.0) or 0.0
        vocab_size = getattr(self, "vocab_size", None)
        if vocab_size is None:
            vocab_size = len(getattr(self, "vocab", set()))
            self.vocab_size = vocab_size

        if n == 1:
            ngram_counts = self.ngram_counts
            total = getattr(self, "total_unigram_count", None)
            if total is None:
                total = sum(ngram_counts.values())
                self.total_unigram_count = total

            log_p = 0.0
            for token in tokens:
                count = ngram_counts.get((token,), 0)
                if k > 0.0:
                    denom = total + k * vocab_size
                    prob = (count + k) / denom if denom > 0 else 0.0
                else:
                    if count == 0 or total == 0:
                        return float("-inf")
                    prob = count / total
                log_p += log_fn(prob) if log_fn is not None else (math.log(prob) * inv_log_base)
            return log_p

        if len(tokens) < n:
            return 0.0

        ngram_counts = self.ngram_counts
        context_counts = self.context_counts
        log_p = 0.0

        if n == 2:
            prev = tokens[0]
            for curr in tokens[1:]:
                denom_count = context_counts.get((prev,), 0)
                num_count = ngram_counts.get((prev, curr), 0)
                if k > 0.0:
                    denom = denom_count + k * vocab_size
                    prob = (num_count + k) / denom if denom > 0 else 0.0
                else:
                    if denom_count == 0:
                        return float("-inf")
                    if num_count == 0:
                        return float("-inf")
                    prob = num_count / denom_count
                log_p += log_fn(prob) if log_fn is not None else (math.log(prob) * inv_log_base)
                prev = curr
            return log_p

        if n == 3:
            if len(tokens) < 2:
                return 0.0

            bigram_counts = getattr(self, "bigram_counts", None)
            bigram_context_counts = getattr(self, "bigram_context_counts", None)

            w0 = tokens[0]
            w1 = tokens[1]
            if bigram_counts is not None and bigram_context_counts is not None:
                denom_count = bigram_context_counts.get((w0,), 0)
                num_count = bigram_counts.get((w0, w1), 0)
                if k > 0.0:
                    denom = denom_count + k * vocab_size
                    prob = (num_count + k) / denom if denom > 0 else 0.0
                else:
                    if denom_count == 0:
                        return float("-inf")
                    if num_count == 0:
                        return float("-inf")
                    prob = num_count / denom_count
                log_p += log_fn(prob) if log_fn is not None else (math.log(prob) * inv_log_base)

            for w2 in tokens[2:]:
                denom_count = context_counts.get((w0, w1), 0)
                num_count = ngram_counts.get((w0, w1, w2), 0)
                if k > 0.0:
                    denom = denom_count + k * vocab_size
                    prob = (num_count + k) / denom if denom > 0 else 0.0
                else:
                    if denom_count == 0:
                        return float("-inf")
                    if num_count == 0:
                        return float("-inf")
                    prob = num_count / denom_count
                log_p += log_fn(prob) if log_fn is not None else (math.log(prob) * inv_log_base)
                w0, w1 = w1, w2
            return log_p

        for end_idx in range(n, len(tokens) + 1):
            context = tuple(tokens[end_idx - n : end_idx - 1])
            denom_count = context_counts.get(context, 0)
            ngram = context + (tokens[end_idx - 1],)
            num_count = ngram_counts.get(ngram, 0)
            if k > 0.0:
                denom = denom_count + k * vocab_size
                prob = (num_count + k) / denom if denom > 0 else 0.0
            else:
                if denom_count == 0:
                    return float("-inf")
                if num_count == 0:
                    return float("-inf")
                prob = num_count / denom_count
            log_p += log_fn(prob) if log_fn is not None else (math.log(prob) * inv_log_base)
        return log_p

    def _log_probability_interpolated(self, tokens: List[Any], log_fn, inv_log_base: float | None):
        n = self.n
        lambdas = self.interpolation_lambdas
        if lambdas is None:
            raise ValueError("Interpolation requested but interpolation_lambdas is None")
        if len(lambdas) != n:
            raise ValueError(f"Expected {n} interpolation lambdas, got {len(lambdas)}")

        if log_fn is None:
            def logp(x: float) -> float:
                return math.log(x) * inv_log_base  # type: ignore[operator]
        else:
            logp = log_fn

        k = getattr(self, "laplace_k", 0.0) or 0.0
        vocab_size = getattr(self, "vocab_size", None)
        if vocab_size is None:
            vocab_size = len(getattr(self, "vocab", set()))
            self.vocab_size = vocab_size

        unigram_counts = getattr(self, "unigram_counts", None)
        total_unigrams = getattr(self, "total_unigram_count", None)
        if unigram_counts is None or total_unigrams is None:
            raise ValueError("Interpolation requires unigram_counts and total_unigram_count (call learn())")

        counts_by_order = getattr(self, "counts_by_order", None)
        context_by_order = getattr(self, "context_counts_by_order", None)

        def p_unigram(word):
            count = unigram_counts.get((word,), 0)
            if k > 0.0:
                denom = total_unigrams + k * vocab_size
                return (count + k) / denom if denom > 0 else 0.0
            return (count / total_unigrams) if total_unigrams > 0 else 0.0

        def p_order(order: int, context: tuple, word):
            if order == 1:
                return p_unigram(word)
            if counts_by_order is None or context_by_order is None:
                # Fallback to legacy fields for n<=3
                if order == 2:
                    bigram_counts = getattr(self, "bigram_counts", None) or getattr(self, "ngram_counts", None)
                    bigram_ctx = getattr(self, "bigram_context_counts", None) or getattr(self, "context_counts", None)
                    num = bigram_counts.get((context[0], word), 0)
                    den = bigram_ctx.get((context[0],), 0)
                elif order == 3:
                    num = self.ngram_counts.get((context[0], context[1], word), 0)
                    den = self.context_counts.get((context[0], context[1]), 0)
                else:
                    return 0.0
            else:
                num = counts_by_order[order].get(context + (word,), 0)
                den = context_by_order[order].get(context, 0)

            if k > 0.0:
                denom = den + k * vocab_size
                return (num + k) / denom if denom > 0 else 0.0
            return (num / den) if den > 0 else 0.0

        log_p_total = 0.0

        # Match the starter/test convention: don't score the first token for n>=2.
        for i in range(1, len(tokens)):
            available_orders = []
            weight_sum = 0.0
            for order in range(1, n + 1):
                if order == 1 or i >= order - 1:
                    w = float(lambdas[order - 1])
                    if w > 0.0:
                        available_orders.append(order)
                        weight_sum += w

            if weight_sum == 0.0:
                return float("-inf")

            prob = 0.0
            for order in available_orders:
                w = float(lambdas[order - 1]) / weight_sum
                context = tuple(tokens[i - (order - 1) : i]) if order > 1 else tuple()
                prob += w * p_order(order, context, tokens[i])

            if prob <= 0.0:
                return float("-inf")
            log_p_total += logp(prob)

        return log_p_total

    def generate(self, num_samples: int, max_steps: int, results_file: str):
        """
        Function for generating text using the n-gram model.

        Inputs:
            num_samples (int): How many samples to generate
            max_steps (int): The maximum length of any sampled output
            results_file (str): Where to save the generated examples
        """
        if num_samples <= 0 or max_steps <= 0:
            return

        if not hasattr(self, "_sampling_tables"):
            self._build_sampling_tables()

        os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)
        rng = random.Random(0)
        samples = []

        for _ in range(num_samples):
            generated = self._generate_one(rng, max_steps)
            samples.append(generated)

        with open(results_file, "w", encoding="utf-8") as f:
            for tokens in samples:
                if tokens and isinstance(tokens[0], str) and len(tokens[0]) == 1:
                    f.write("".join(tokens) + "\n")
                else:
                    f.write(" ".join(map(str, tokens)) + "\n")

    def _build_sampling_tables(self):
        n = self.n
        ngram_counts = self.ngram_counts

        if n == 1:
            tokens = []
            cum_weights = []
            total = 0
            for (token,), count in ngram_counts.items():
                if count <= 0:
                    continue
                total += count
                tokens.append(token)
                cum_weights.append(total)
            self._sampling_tables = {"unigram": (tokens, cum_weights, total)}
            return

        context_to_counts = {}
        for ngram, count in ngram_counts.items():
            if count <= 0:
                continue
            context = ngram[:-1]
            next_token = ngram[-1]
            ctx_counts = context_to_counts.get(context)
            if ctx_counts is None:
                ctx_counts = {}
                context_to_counts[context] = ctx_counts
            ctx_counts[next_token] = ctx_counts.get(next_token, 0) + count

        tables = {}
        for context, next_counts in context_to_counts.items():
            next_tokens = []
            cum_weights = []
            total = 0
            for token, count in next_counts.items():
                total += count
                next_tokens.append(token)
                cum_weights.append(total)
            tables[context] = (next_tokens, cum_weights, total)

        contexts = list(tables.keys())
        self._sampling_tables = {"contexts": contexts, "tables": tables}

    def _sample_from_table(self, rng: random.Random, table):
        tokens, cum_weights, total = table
        if total <= 0:
            return None
        r = rng.randint(1, total)
        idx = bisect.bisect_left(cum_weights, r)
        return tokens[idx]

    def _generate_one(self, rng: random.Random, max_steps: int):
        n = self.n
        if n == 1:
            table = self._sampling_tables["unigram"]
            return [self._sample_from_table(rng, table) for _ in range(max_steps)]

        tables = self._sampling_tables["tables"]
        contexts = self._sampling_tables["contexts"]
        if not contexts:
            return []

        context = contexts[rng.randrange(len(contexts))]
        generated = list(context)
        while len(generated) < max_steps:
            table = tables.get(tuple(generated[-(n - 1) :]))
            if table is None:
                context = contexts[rng.randrange(len(contexts))]
                generated.extend(list(context))
                generated = generated[:max_steps]
                continue

            next_token = self._sample_from_table(rng, table)
            if next_token is None:
                break
            generated.append(next_token)
        return generated[:max_steps]

    def learn(self, training_data: List[List[Any]]):
        """
        Function for learning n-grams from the provided training data. You may
        add keywords to this function as needed, provided that the default behavior
        is that of an n-gram model without any label smoothing.
        
        Inputs:
            training_data (List[List[Any]]): A list of model inputs, which should each be lists
                                             of input tokens
        """
        self.unigram_counts = Counter()
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()
        self.bigram_counts = Counter() if self.n >= 3 else None
        self.bigram_context_counts = Counter() if self.n >= 3 else None
        self.counts_by_order = None
        self.context_counts_by_order = None

        build_all_orders = (self.interpolation_lambdas is not None and self.n > 3)
        if build_all_orders:
            self.counts_by_order = [None] * (self.n + 1)
            self.context_counts_by_order = [None] * (self.n + 1)
            for order in range(2, self.n + 1):
                self.counts_by_order[order] = Counter()
                self.context_counts_by_order[order] = Counter()

        for sentence in training_data:
            self.vocab.update(sentence)

            for token in sentence:
                self.unigram_counts[(token,)] += 1

            if build_all_orders:
                for order in range(2, self.n + 1):
                    if len(sentence) < order:
                        break
                    counts = self.counts_by_order[order]
                    contexts = self.context_counts_by_order[order]
                    for start_idx in range(len(sentence) - order + 1):
                        ngram = tuple(sentence[start_idx:start_idx + order])
                        counts[ngram] += 1
                        contexts[ngram[:-1]] += 1
                continue

            if self.n >= 3 and len(sentence) >= 2:
                prev = sentence[0]
                for curr in sentence[1:]:
                    self.bigram_counts[(prev, curr)] += 1
                    self.bigram_context_counts[(prev,)] += 1
                    prev = curr

            if self.n == 1:
                for token in sentence:
                    self.ngram_counts[(token,)] += 1
                continue

            if len(sentence) < self.n:
                continue

            for start_idx in range(len(sentence) - self.n + 1):
                ngram = tuple(sentence[start_idx:start_idx + self.n])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

        if self.n == 1:
            self.total_unigram_count = sum(self.ngram_counts.values())
        else:
            self.total_unigram_count = sum(self.unigram_counts.values())

        if build_all_orders:
            self.ngram_counts = self.counts_by_order[self.n]
            self.context_counts = self.context_counts_by_order[self.n]
            if self.n >= 3:
                self.bigram_counts = self.counts_by_order[2]
                self.bigram_context_counts = self.context_counts_by_order[2]

        self.vocab.add(self.unk_token)
        self.vocab_size = len(self.vocab)
        if hasattr(self, "_sampling_tables"):
            delattr(self, "_sampling_tables")

        
def main():
    # Get key arguments
    args = get_args()

    # Get the data for language-modeling and WER computation
    tokenization_level = args.tokenization_level
    train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data = load_data(tokenization_level) # TODO
    # Initialize and "train" the n-gram model
    n = args.n
    use_interpolation = not args.no_interpolate and n >= 2
    interp_lambdas = None
    if use_interpolation:
        if (args.interp_lambdas or "").strip().lower() == "auto":
            level = (tokenization_level or "").strip().lower()
            if "char" in level:
                # Geometric weighting towards higher orders: [1, 2, 4, ..., 2^(n-1)] normalized.
                weights = [2 ** i for i in range(n)]
            else:
                # Uniform weights are a safer default for word/subword where sparsity is higher.
                weights = [1.0 for _ in range(n)]
            s = float(sum(weights))
            interp_lambdas = [w / s for w in weights]
        else:
            parts = [p.strip() for p in args.interp_lambdas.split(",") if p.strip()]
            interp_lambdas = [float(p) for p in parts]
            if len(interp_lambdas) != n:
                raise ValueError(f"--interp_lambdas must have {n} values (got {len(interp_lambdas)})")
            s = float(sum(interp_lambdas))
            if s <= 0.0:
                raise ValueError("--interp_lambdas must sum to a positive value")
            interp_lambdas = [x / s for x in interp_lambdas]

    model = NGramLM(n, laplace_k=args.laplace_k, interpolation_lambdas=interp_lambdas)
    model.learn(train_data)

    # Evaluate model perplexity
    val_perplexity = evaluate_perplexity(model, val_data)
    print(f'Model perplexity on the val set: {val_perplexity}')
    dev_perplexity = evaluate_perplexity(model, dev_data)
    print(f'Model perplexity on the dev set: {dev_perplexity}')
    test_perplexity = evaluate_perplexity(model, test_data)
    print(f'Model perplexity on the test set: {test_perplexity}')    

    # Evaluate model WER
    experiment_name = args.experiment_name
    dev_wer_savepath = os.path.join('results', f'{experiment_name}_n_gram_dev_wer_predictions.csv')
    rerank_sentences_for_wer(model, dev_wer_data, dev_wer_savepath)
    dev_wer = compute_wer('data/wer_data/dev_ground_truths.csv', dev_wer_savepath)
    print("Dev set WER was: ", dev_wer)

    test_wer_savepath = os.path.join('results', f'{experiment_name}_n_gram_test_wer_predictions.csv')
    rerank_sentences_for_wer(model, test_wer_data, test_wer_savepath)

    # Generate text from the model
    generation_path = os.path.join('generations', f'{experiment_name}_n_gram_generation_examples.pkl')
    num_samples = args.num_samples
    max_steps = args.max_steps
    model.generate(num_samples, max_steps, generation_path)
    

if __name__ == "__main__":
    main()
    
