"""Hermetic test for per-clip dedup of reranked windows."""
from das.retrieval import dedupe_by_clip

WINDOWS = [
    {"clip_id": "a"}, {"clip_id": "a"}, {"clip_id": "b"},
    {"clip_id": "c"}, {"clip_id": "b"},
]


def test_keeps_best_window_per_clip_in_rank_order():
    ranked = [(1, 0.9), (0, 0.8), (4, 0.7), (2, 0.6), (3, 0.5)]
    out = dedupe_by_clip(ranked, WINDOWS, k=5)
    assert out == [(1, 0.9), (4, 0.7), (3, 0.5)]  # one per clip: a, b, c


def test_stops_at_k():
    ranked = [(1, 0.9), (2, 0.8), (3, 0.7)]
    assert dedupe_by_clip(ranked, WINDOWS, k=2) == [(1, 0.9), (2, 0.8)]
