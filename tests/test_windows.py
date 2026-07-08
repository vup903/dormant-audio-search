"""Hermetic tests for segment->window grouping."""
from das.windows import build_windows


def seg(start, end, text="hello", conf=0.8):
    return {"start": start, "end": end, "text": text, "confidence": conf}


def test_groups_until_target():
    segments = [seg(i * 10, i * 10 + 10) for i in range(10)]  # 100s of 10s segments
    windows = build_windows(segments, target_seconds=45.0)
    assert len(windows) == 2
    assert windows[0]["start"] == 0
    assert windows[0]["end"] == 50  # 5 segments reach the 45s target
    assert windows[1]["end"] == 100


def test_trailing_partial_window_kept():
    segments = [seg(0, 30), seg(30, 60), seg(60, 70)]
    windows = build_windows(segments, target_seconds=45.0)
    assert len(windows) == 2
    assert windows[1]["text"] == "hello"


def test_confidence_is_duration_weighted():
    segments = [seg(0, 40, conf=1.0), seg(40, 50, conf=0.0)]
    windows = build_windows(segments, target_seconds=45.0)
    assert len(windows) == 1
    assert abs(windows[0]["confidence"] - 0.8) < 1e-6


def test_empty_text_dropped():
    assert build_windows([seg(0, 10, text="  ")]) == []
