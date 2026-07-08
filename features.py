from collections import ChainMap
from typing import Callable, Dict, Set

import pandas as pd


class FeatureMap:
    name: str

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        pass

    @classmethod
    def prefix_with_name(self, d: Dict) -> Dict[str, float]:
        """just a handy shared util function"""
        return {f"{self.name}/{k}": v for k, v in d.items()}


class BagOfWords(FeatureMap):
    name = "bow"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        tokens = [t.lower() for t in text.split()]
        ret: Dict[str, float] = {}

        for token in tokens:
            if token in self.STOP_WORDS:
                continue
            ret[token] = 1.0

        return self.prefix_with_name(ret)

class SentenceLength(FeatureMap):
    name = "len"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """an example of custom feature that rewards long sentences"""
        if len(text.split()) < 10:
            k = "short"
            v = 1.0
        else:
            k = "long"
            v = 5.0
        ret = {k: v}
        return self.prefix_with_name(ret)


class BagOfBigrams(FeatureMap):
    name = "bigram"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        tokens = text.split()
        non_stop = [t for t in tokens if t not in self.STOP_WORDS]
        ret: Dict[str, float] = {}
        for i in range(len(non_stop) - 1):
            bigram = f"{non_stop[i]}_{non_stop[i+1]}"
            ret[bigram] = ret.get(bigram, 0) + 1
        return self.prefix_with_name(ret)


class BagOfTrigrams(FeatureMap):
    name = "trigram"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        tokens = text.split()
        non_stop = [t for t in tokens if t not in self.STOP_WORDS]
        ret: Dict[str, float] = {}
        for i in range(len(non_stop) - 2):
            trigram = f"{non_stop[i]}_{non_stop[i+1]}_{non_stop[i+2]}"
            ret[trigram] = ret.get(trigram, 0) + 1
        return self.prefix_with_name(ret)

class SubjectTokens(FeatureMap):
    """Bag of words from the Subject line (or first line if no Subject:). For Newsgroups, Subject is highly discriminative."""
    name = "subj"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])
    SUBJECT_PREFIX = "subject:"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        lines = text.split("\n")
        subject_line = ""
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith(self.SUBJECT_PREFIX):
                subject_line = stripped[len(self.SUBJECT_PREFIX) :].strip()
                break
        if not subject_line and lines:
            subject_line = lines[0].strip()
        tokens = [t.lower() for t in subject_line.split()]
        ret: Dict[str, float] = {}
        for token in tokens:
            if token in self.STOP_WORDS:
                continue
            ret[token] = ret.get(token, 0) + 1
        return self.prefix_with_name(ret)


class SentimentLexicon(FeatureMap):
    name = "lex"
    POSITIVE = {
        "good", "great", "love", "best", "amazing", "excellent", "wonderful",
        "fantastic", "beautiful", "perfect", "enjoy", "enjoyed", "fun", "funny",
        "brilliant", "awesome", "loved", "liked", "like", "better", "positive",
        "happy", "pleased", "impressed", "recommend", "worth", "outstanding",
        "superb", "remarkable", "delightful", "charming", "engaging", "touching",
        "inspiring", "moving", "powerful", "masterpiece", "gem", "treasure",
        "joy", "joyful", "heartwarming", "uplifting", "refreshing", "solid",
        "strong", "compelling", "captivating", "entertaining", "satisfying",
        "memorable", "unforgettable", "flawless", "stellar", "top", "favorite",
        "admire", "admirable", "praise", "praised", "success", "successful",
        "win", "winner", "won", "triumph", "triumphant", "magic", "magical",
    }
    NEGATIVE = {
        "bad", "terrible", "hate", "worst", "awful", "boring", "waste",
        "horrible", "stupid", "dull", "poor", "weak", "disappointing", "failed",
        "hated", "dislike", "worse", "negative", "angry", "annoying", "ridiculous",
        "pathetic", "mess", "garbage", "vile", "dreadful", "miserable", "sucks",
        "suck", "unbearable", "unwatchable", "pointless", "useless", "lame",
        "cheap", "lazy", "confusing", "predictable", "forgettable", "generic",
        "flat", "empty", "hollow", "pretentious", "overrated", "underwhelming",
        "frustrating", "irritating", "tedious", "painful", "cringe", "cringey",
        "disaster", "disastrous", "failure", "flawed", "broken", "ruined",
        "ruin", "destroy", "destroyed", "trash", "junk", "mediocre", "bland",
        "silly", "dumb", "nonsense", "absurd", "offensive", "disturbing",
        "depressing", "sad", "frustrated", "confused", "disappointed", "annoyed",
    }

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        tokens = [t.lower() for t in text.split()]
        pos_count = sum(1 for t in tokens if t in self.POSITIVE)
        neg_count = sum(1 for t in tokens if t in self.NEGATIVE)
        return self.prefix_with_name({"pos": float(pos_count), "neg": float(neg_count)})


class NegationAware(FeatureMap):
    name = "neg"
    NEGATION_TRIGGERS = {"not", "n't", "never", "no", "neither", "nor"}
    NEGATION_WINDOW = 2
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        tokens = text.split()
        lower = [t.lower() for t in tokens]
        negated_indices: Set[int] = set()
        for i, tok in enumerate(lower):
            if tok in self.NEGATION_TRIGGERS:
                for j in range(1, self.NEGATION_WINDOW + 1):
                    if i + j < len(tokens):
                        negated_indices.add(i + j)
        ret: Dict[str, float] = {}
        pos_neg, neg_neg = 0.0, 0.0
        for i in negated_indices:
            t = tokens[i]
            tl = lower[i]
            if tl not in self.STOP_WORDS:
                key = f"{tl}_neg"
                ret[key] = ret.get(key, 0) + 1
            if tl in SentimentLexicon.POSITIVE:
                pos_neg += 1
            if tl in SentimentLexicon.NEGATIVE:
                neg_neg += 1
        ret["pos_neg"] = pos_neg
        ret["neg_neg"] = neg_neg
        return self.prefix_with_name(ret)


FEATURE_CLASSES_MAP = {c.name: c for c in [BagOfWords, SentenceLength, BagOfBigrams, BagOfTrigrams, SubjectTokens, SentimentLexicon, NegationAware]}


def make_featurize(
    feature_types: Set[str],
) -> Callable[[str], Dict[str, float]]:
    featurize_fns = [FEATURE_CLASSES_MAP[n].featurize for n in feature_types]

    def _featurize(text: str):
        f = ChainMap(*[fn(text) for fn in featurize_fns])
        return dict(f)

    return _featurize


__all__ = ["make_featurize"]

if __name__ == "__main__":
    text = "I love this movie"
    print(text)
    print(BagOfWords.featurize(text))
    featurize = make_featurize({"bow", "len"})
    print(featurize(text))
