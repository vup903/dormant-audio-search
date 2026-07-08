"""Group ASR segments into ~45-second windows.

Windows are the retrieval unit: big enough to carry meaning, small enough that
the timecode drops a listener at the right moment. Shared by tagging and
indexing so both see the same text.
"""
from __future__ import annotations

TARGET_SECONDS = 45.0


def _merge(segments: list[dict]) -> dict:
    total = sum(s["end"] - s["start"] for s in segments) or 1.0
    confidence = sum((s["end"] - s["start"]) * s["confidence"] for s in segments) / total
    return {
        "start": segments[0]["start"],
        "end": segments[-1]["end"],
        "text": " ".join(s["text"] for s in segments),
        "confidence": round(confidence, 4),
    }


def build_windows(segments: list[dict], target_seconds: float = TARGET_SECONDS) -> list[dict]:
    windows: list[dict] = []
    current: list[dict] = []
    for seg in segments:
        if not seg["text"].strip():
            continue
        current.append(seg)
        if current[-1]["end"] - current[0]["start"] >= target_seconds:
            windows.append(_merge(current))
            current = []
    if current:
        windows.append(_merge(current))
    return windows
