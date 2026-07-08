"""Fetch public-domain historical radio broadcasts from archive.org.

Selection is deterministic (sorted, evenly spaced across each source item) so
reruns are reproducible, and downloads are idempotent (existing files are
skipped). Every clip records its source_url and a rights statement.

No NPR audio is used anywhere in this project.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "data" / "audio"
MANIFEST = ROOT / "data" / "manifest.json"

# Curated archive.org items (all from the same uploader's "Radio News" series),
# chosen to cover three decades. quota = how many clips to sample per item.
SOURCES = [
    {"item": "1938RadioNews", "decade": "1930s", "quota": 5},
    {"item": "1939RadioNews", "decade": "1930s", "quota": 7},
    {"item": "1941RadioNews", "decade": "1940s", "quota": 6},
    {"item": "1944RadioNews", "decade": "1940s", "quota": 6},
    {"item": "1945RadioNews", "decade": "1940s", "quota": 5},
    {"item": "1948RadioNews", "decade": "1940s", "quota": 5},
    {"item": "1950-1959RadioNews", "decade": "1950s", "quota": 12},
]

# These 1930s-50s U.S. broadcasts carry no explicit rights statement on
# archive.org; record that honestly instead of inventing a clean label.
RIGHTS = "public domain (presumed; pre-1963 broadcast, no rights statement on archive.org item)"

MIN_SECONDS = 60
MAX_SECONDS = 32 * 60

FILENAME_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-([A-Za-z]+)-(.+)\.mp3$")


def parse_length(value) -> float | None:
    """archive.org 'length' is either seconds ('311.4') or 'MM:SS'/'HH:MM:SS'."""
    if value is None:
        return None
    s = str(value)
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        seconds = 0.0
        for p in parts:
            seconds = seconds * 60 + p
        return seconds
    try:
        return float(s)
    except ValueError:
        return None


def parse_filename(name: str) -> dict | None:
    m = FILENAME_RE.match(name)
    if not m:
        return None
    year, month, day, network, rest = m.groups()
    title = re.sub(r"[-_]+", " ", rest).strip()
    return {
        "date": f"{year}-{month}-{day}",
        "program": network.upper(),
        "title": title,
    }


def evenly_spaced(items: list, k: int) -> list:
    """Pick k items spread across the (sorted) list — deterministic sampling."""
    n = len(items)
    if n <= k:
        return list(items)
    if k == 1:
        return [items[0]]
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


def select_clips(source: dict, session: requests.Session) -> list[dict]:
    item = source["item"]
    meta = session.get(f"https://archive.org/metadata/{item}", timeout=60).json()
    candidates = []
    for f in meta.get("files", []):
        name = f.get("name", "")
        if not name.lower().endswith(".mp3"):
            continue
        parsed = parse_filename(name)
        length = parse_length(f.get("length"))
        if parsed is None or length is None:
            continue
        if not (MIN_SECONDS <= length <= MAX_SECONDS):
            continue
        candidates.append({
            "id": Path(name).stem.lower(),
            "item": item,
            "filename": name,
            "decade": source["decade"],
            "duration_s": round(length, 1),
            "size_bytes": int(f.get("size", 0)),
            "source_url": f"https://archive.org/details/{item}",
            "file_url": f"https://archive.org/download/{item}/{name}",
            "rights": RIGHTS,
            **parsed,
        })
    candidates.sort(key=lambda c: c["filename"])
    picked = evenly_spaced(candidates, source["quota"])
    print(f"  {item}: {len(candidates)} usable mp3s -> picked {len(picked)}")
    return picked


def download(clip: dict, session: requests.Session) -> Path:
    dest = AUDIO_DIR / f"{clip['id']}.mp3"
    if dest.exists() and dest.stat().st_size == clip["size_bytes"]:
        print(f"  [skip] {dest.name}")
        return dest
    print(f"  [get ] {clip['filename']} ({clip['duration_s'] / 60:.1f} min)")
    with session.get(clip["file_url"], stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
        tmp.replace(dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="download at most N clips total (for smoke tests)")
    ap.add_argument("--dry-run", action="store_true", help="select only, no downloads")
    args = ap.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "dormant-audio-search/0.1 (research prototype)"

    print("Selecting clips:")
    clips: list[dict] = []
    for source in SOURCES:
        clips.extend(select_clips(source, session))

    ids = [c["id"] for c in clips]
    assert len(ids) == len(set(ids)), "clip id collision — filenames not unique"

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(clips, indent=2), encoding="utf-8")
    total_min = sum(c["duration_s"] for c in clips) / 60
    print(f"Manifest: {len(clips)} clips, {total_min:.0f} min total -> {MANIFEST}")

    if args.dry_run:
        return 0

    to_download = clips[: args.limit] if args.limit else clips
    print("Downloading (sequential, idempotent):")
    for clip in to_download:
        download(clip, session)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
