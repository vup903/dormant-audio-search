"""Hand-built BM25 (Okapi) — the lexical half of two-stage retrieval.

~50 lines, no framework. Lexical scoring catches exact words (names, places,
programs) that embeddings blur; the semantic stage catches paraphrase the
lexicon misses. Neither alone is enough on noisy ASR text.
"""
from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if docs else 0.0
        self.doc_tf = [Counter(t) for t in self.doc_tokens]
        df: Counter = Counter()
        for tf in self.doc_tf:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {
            term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()
        }

    def score(self, query: str, doc_index: int) -> float:
        tf = self.doc_tf[doc_index]
        dl = self.doc_len[doc_index] or 1
        score = 0.0
        for term in tokenize(query):
            if term not in tf:
                continue
            f = tf[term]
            score += self.idf.get(term, 0.0) * (
                f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg_len))
            )
        return score

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(len(self.doc_tokens))]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
