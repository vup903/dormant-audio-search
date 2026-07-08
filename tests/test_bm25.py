"""Hermetic tests for the hand-built BM25."""
from das.bm25 import BM25, tokenize


DOCS = [
    "allied forces landed on the beaches of normandy this morning",
    "the president spoke about the economy and rising prices",
    "normandy invasion continues as allied troops push inland",
    "baseball scores from yesterday's games across the league",
]


def test_tokenize_lowercases_and_splits():
    assert tokenize("D-Day: Allied FORCES!") == ["d", "day", "allied", "forces"]


def test_relevant_docs_rank_first():
    bm25 = BM25(DOCS)
    top = bm25.top("allied normandy landing", k=4)
    top_ids = [i for i, _ in top]
    assert set(top_ids[:2]) == {0, 2}


def test_no_match_returns_empty():
    bm25 = BM25(DOCS)
    assert bm25.top("zeppelin", k=4) == []


def test_rare_term_outweighs_common():
    docs = ["war war war economy", "normandy report", "war news today"]
    bm25 = BM25(docs)
    top = bm25.top("normandy", k=3)
    assert top[0][0] == 1
