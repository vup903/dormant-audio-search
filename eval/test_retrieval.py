"""Retrieval evaluation harness (needs the index + models — not hermetic).

Runs every golden query through the real two-stage search and scores hit@5:
did any expected clip appear in the top 5? Scores are broken down per decade —
that breakdown is the bias evaluation: it shows WHO the tool fails, not just
how often.

Writes eval/results.json (rendered at /eval) and fails the suite if overall
hit@5 drops below the floor — quality can't quietly rot.

Run: python -m pytest eval/ -q
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).parent / "golden.yaml"
RESULTS = Path(__file__).parent / "results.json"

HIT_AT_5_FLOOR = 0.8


@pytest.fixture(scope="session")
def searcher():
    from das.retrieval import Searcher
    return Searcher()


@pytest.fixture(scope="session")
def golden() -> list[dict]:
    return yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["queries"]


def test_hit_at_5(searcher, golden):
    rows = []
    for case in golden:
        results = searcher.search(case["query"], k=5)
        got = [r["clip_id"] for r in results]
        hit = any(clip in got for clip in case["expected"])
        rows.append({
            "query": case["query"],
            "decade": case["decade"],
            "expected": case["expected"],
            "got": got,
            "hit": hit,
        })

    decades = sorted({r["decade"] for r in rows})
    per_decade = {}
    for d in decades:
        sub = [r for r in rows if r["decade"] == d]
        per_decade[d] = {
            "hits": sum(r["hit"] for r in sub),
            "total": len(sub),
            "hit_at_5": round(sum(r["hit"] for r in sub) / len(sub), 3),
        }
    overall = round(sum(r["hit"] for r in rows) / len(rows), 3)

    RESULTS.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric": "hit@5 (any expected clip in top 5)",
        "retrieval": "semantic ∪ lexical top-8 → cross-encoder rerank → top-5",
        "overall": {"hits": sum(r["hit"] for r in rows), "total": len(rows), "hit_at_5": overall},
        "per_decade": per_decade,
        "queries": rows,
    }, indent=2), encoding="utf-8")

    print(f"\noverall hit@5 = {overall}  " +
          "  ".join(f"{d}: {v['hit_at_5']}" for d, v in per_decade.items()))
    assert overall >= HIT_AT_5_FLOOR, (
        f"hit@5 {overall} fell below the {HIT_AT_5_FLOOR} floor — "
        "see eval/results.json for which queries broke"
    )
