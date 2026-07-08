"""Build one schema-validated JSON record per clip.

Merges manifest metadata with the transcript, then assigns topic tags by
zero-shot embedding similarity against the controlled vocabulary — the model
can only ever emit terms from vocab.json, and the schema rejects anything
else. Every AI-produced field is marked ai_generated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema

from das.windows import build_windows

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.json"
TRANSCRIPT_DIR = ROOT / "data" / "transcripts"
RECORD_DIR = ROOT / "data" / "records"
VOCAB = ROOT / "vocab.json"
SCHEMA = ROOT / "schema" / "record.schema.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
TAG_THRESHOLD = 0.28
MAX_TAGS = 3


def tag_scores(model, windows: list[dict], terms: dict[str, str]):
    """Score each vocab term against the clip: max cosine over windows."""
    import numpy as np

    term_texts = [f"{term}: {desc}" for term, desc in terms.items()]
    term_vecs = model.encode(term_texts, normalize_embeddings=True)
    win_vecs = model.encode([w["text"] for w in windows], normalize_embeddings=True)
    sims = win_vecs @ term_vecs.T  # (windows, terms)
    best = sims.max(axis=0)
    return dict(zip(terms.keys(), (float(x) for x in best)))


def pick_tags(scores: dict[str, float]) -> list[dict]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    picked = [(t, s) for t, s in ranked[:MAX_TAGS] if s >= TAG_THRESHOLD]
    if not picked:  # always keep the single best guess, low score and all
        picked = ranked[:1]
    return [
        {"term": t, "score": round(max(0.0, min(1.0, s)), 4), "ai_generated": True}
        for t, s in picked
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show-scores", action="store_true", help="print tag score details")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer  # deferred: heavy import

    clips = json.loads(MANIFEST.read_text(encoding="utf-8"))
    terms = json.loads(VOCAB.read_text(encoding="utf-8"))["terms"]
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL)
    built = 0
    for clip in clips:
        tpath = TRANSCRIPT_DIR / f"{clip['id']}.json"
        if not tpath.exists():
            continue
        transcript = json.loads(tpath.read_text(encoding="utf-8"))
        windows = build_windows(transcript["segments"])
        if not windows:
            print(f"  [skip] {clip['id']}: no usable windows")
            continue

        scores = tag_scores(model, windows, terms)
        tags = pick_tags(scores)
        if args.show_scores:
            top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
            print(f"  {clip['id'][:55]}: " + ", ".join(f"{t}={s:.2f}" for t, s in top))

        record = {
            "id": clip["id"],
            "title": clip["title"],
            "program": clip["program"],
            "date": clip["date"],
            "decade": clip["decade"],
            "source_url": clip["source_url"],
            "rights": clip["rights"],
            "audio_file": f"{clip['id']}.mp3",
            "language": transcript["language"],
            "duration_s": clip["duration_s"],
            "asr": {
                "model": transcript["model"],
                "confidence": transcript["confidence"],
                "ai_generated": True,
            },
            "transcript": " ".join(s["text"] for s in transcript["segments"]),
            "segments": transcript["segments"],
            "tags": tags,
        }
        jsonschema.validate(record, schema)
        (RECORD_DIR / f"{clip['id']}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        built += 1

    print(f"Built {built} validated records -> {RECORD_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
