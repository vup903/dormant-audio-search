"""Transcribe fetched audio with faster-whisper (local, CPU int8).

Idempotent: clips with an existing transcript JSON are skipped, so the batch
can be interrupted and rerun. Keeps per-segment timestamps and confidence —
provenance for every downstream field.

Confidence = exp(avg_logprob) per segment; per-clip value is the
duration-weighted mean. On 1930s-50s recordings this is expected to be low —
surfacing that honestly is the point, not a bug.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "data" / "audio"
TRANSCRIPT_DIR = ROOT / "data" / "transcripts"
MANIFEST = ROOT / "data" / "manifest.json"

MODEL_NAME = "base"


def segment_confidence(avg_logprob: float) -> float:
    return round(min(1.0, max(0.0, math.exp(avg_logprob))), 4)


def transcribe_clip(model, clip: dict) -> dict | None:
    audio = AUDIO_DIR / f"{clip['id']}.mp3"
    if not audio.exists():
        print(f"  [miss] no audio for {clip['id']}")
        return None

    t0 = time.time()
    # condition_on_previous_text=False: noisy shellac-era audio makes Whisper
    # loop/hallucinate when it conditions on its own previous output.
    segments_iter, info = model.transcribe(
        str(audio), beam_size=5, condition_on_previous_text=False
    )
    segments = [
        {
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "text": s.text.strip(),
            "confidence": segment_confidence(s.avg_logprob),
        }
        for s in segments_iter
        if s.text.strip()
    ]
    if not segments:
        print(f"  [warn] empty transcript for {clip['id']}")
        return None

    total = sum(s["end"] - s["start"] for s in segments) or 1.0
    weighted = sum((s["end"] - s["start"]) * s["confidence"] for s in segments) / total
    elapsed = time.time() - t0
    speed = clip["duration_s"] / elapsed if elapsed else 0
    print(
        f"  [done] {clip['id'][:60]}  lang={info.language} "
        f"conf={weighted:.2f} ({elapsed:.0f}s, {speed:.1f}x realtime)"
    )
    return {
        "id": clip["id"],
        "model": f"faster-whisper-{MODEL_NAME}",
        "language": info.language,
        "duration_s": clip["duration_s"],
        "confidence": round(weighted, 4),
        "segments": segments,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="transcribe at most N clips")
    args = ap.parse_args()

    from faster_whisper import WhisperModel  # deferred: heavy import

    clips = json.loads(MANIFEST.read_text(encoding="utf-8"))
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    pending = [
        c for c in clips
        if not (TRANSCRIPT_DIR / f"{c['id']}.json").exists()
        and (AUDIO_DIR / f"{c['id']}.mp3").exists()
    ]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(pending)} clips to transcribe (model={MODEL_NAME}, cpu int8)")

    model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    for clip in pending:
        result = transcribe_clip(model, clip)
        if result:
            out = TRANSCRIPT_DIR / f"{clip['id']}.json"
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
