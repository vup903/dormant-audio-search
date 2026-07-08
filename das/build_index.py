"""Build the search index: 45s windows + their embeddings.

Persists two artifacts under data/index/:
  windows.json   — retrieval units with clip_id, timecodes, text, confidence
  embeddings.npy — one normalized vector per window (numpy, no vector DB)

BM25 is rebuilt in memory at load time from windows.json — the corpus is small
and it keeps the index format to two files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from das.windows import build_windows

ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = ROOT / "data" / "records"
INDEX_DIR = ROOT / "data" / "index"

EMBED_MODEL = "all-MiniLM-L6-v2"


def main() -> int:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    records = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(RECORD_DIR.glob("*.json"))
    ]
    if not records:
        print("No records found — run build_records first.")
        return 1

    windows = []
    for rec in records:
        for w in build_windows(rec["segments"]):
            windows.append({
                "clip_id": rec["id"],
                "start": w["start"],
                "end": w["end"],
                "text": w["text"],
                # what search sees: the recording's own metadata travels with
                # every window (speakers rarely say their own name or the date)
                "index_text": f"{rec['title']}. {rec['program']}, {rec['date']}. {w['text']}",
                "confidence": w["confidence"],
            })

    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode(
        [w["index_text"] for w in windows],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "windows.json").write_text(json.dumps(windows, indent=2), encoding="utf-8")
    np.save(INDEX_DIR / "embeddings.npy", vecs)
    print(f"Indexed {len(windows)} windows from {len(records)} records -> {INDEX_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
