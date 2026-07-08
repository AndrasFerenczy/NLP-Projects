from typing import List, Any, Tuple
import os
import pandas as pd
import torch


def _edit_distance(ref_tokens: List[str], hyp_tokens: List[str]) -> int:
    if not ref_tokens:
        return 0 if not hyp_tokens else len(hyp_tokens)
    if not hyp_tokens:
        return len(ref_tokens)

    prev = list(range(len(hyp_tokens) + 1))
    for i, ref_tok in enumerate(ref_tokens, start=1):
        curr = [i]
        for j, hyp_tok in enumerate(hyp_tokens, start=1):
            cost = 0 if ref_tok == hyp_tok else 1
            curr.append(min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost  # substitution
            ))
        prev = curr
    return prev[-1]


def _corpus_wer(references: List[str], predictions: List[str]) -> float:
    total_edits = 0
    total_ref_words = 0
    for ref, hyp in zip(references, predictions):
        ref_tokens = str(ref).split()
        hyp_tokens = str(hyp).split()
        total_edits += _edit_distance(ref_tokens, hyp_tokens)
        total_ref_words += len(ref_tokens)
    return (total_edits / total_ref_words) if total_ref_words > 0 else 0.0

def rerank_sentences_for_wer(model: Any, wer_data: List[Any], savepath: str):
    """
    Function to rerank candidate sentences in the HUB dataset. For each set of sentences,
    you must assign each sentence a score in the form of the sentence's acoustic score plus
    the sentence's log probability. You should then save the top scoring sentences in a .csv
    file similar to those found in the results directory.

    Inputs:
        model (Any): An n-gram or Transformer model.
        wer_data (List[Any]): Processed data from the HUB dataset. 
        savepath (str): The path to save the csv file pairing sentence set ids and the top ranked sentences.
    """
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)

    rows = []
    for item in wer_data:
        utterance_id = item["id"]
        sentences = item["sentences"]
        tokenized_sentences = item["tokenized_sentences"]
        acoustic_scores = item["acoustic_scores"]

        best_score = float("-inf")
        best_sentence = sentences[0] if sentences else ""

        for sentence, tokenized_sentence, acoustic_score in zip(sentences, tokenized_sentences, acoustic_scores):
            try:
                lm_score = model.log_probability(tokenized_sentence)
            except TypeError as e:
                raise TypeError(
                    "model.log_probability() signature not supported by rerank_sentences_for_wer(); "
                    "expected an n-gram style log_probability(tokens) implementation."
                ) from e

            total_score = float(acoustic_score) + float(lm_score)
            if total_score > best_score:
                best_score = total_score
                best_sentence = sentence

        rows.append({"id": utterance_id, "sentences": best_sentence})

    pd.DataFrame(rows).to_csv(savepath, index=False)

def compute_wer(gt_path, model_path):
    # Load the sentences
    ground_truths = pd.read_csv(gt_path)['sentences'].tolist()
    guesses = pd.read_csv(model_path)['sentences'].tolist()

    # Compute WER (prefer `evaluate` if present, otherwise fall back to a local implementation)
    try:
        from evaluate import load  # type: ignore
    except ModuleNotFoundError:
        return _corpus_wer(ground_truths, guesses)

    try:
        wer = load("wer")
        return wer.compute(predictions=guesses, references=ground_truths)
    except (PermissionError, OSError):
        # In sandboxed environments we may be unable to create HuggingFace cache lockfiles.
        # Fall back to a local WER implementation.
        return _corpus_wer(ground_truths, guesses)
