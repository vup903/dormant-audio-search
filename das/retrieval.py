"""Two-stage retrieval, hand-built (same shape as personal-rag).

Stage 1 — candidates: semantic top-8 (cosine over MiniLM embeddings)
                      ∪ lexical top-8 (hand-built BM25)
Stage 2 — rerank:     cross-encoder scores every candidate against the query,
                      keep top 5.

Results carry everything a reader needs to verify: excerpt, program, date,
timecode, ASR confidence, tags (marked AI-generated) and the archive.org
source link. No index entry, no answer — results only ever come from the corpus.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from das.bm25 import BM25

ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = ROOT / "data" / "records"
INDEX_DIR = ROOT / "data" / "index"

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES_PER_STAGE = 8
TOP_K = 5


class Searcher:
    def __init__(self) -> None:
        import numpy as np
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.windows = json.loads((INDEX_DIR / "windows.json").read_text(encoding="utf-8"))
        self.embeddings = np.load(INDEX_DIR / "embeddings.npy")
        self.records = {
            r["id"]: r
            for r in (
                json.loads(p.read_text(encoding="utf-8"))
                for p in RECORD_DIR.glob("*.json")
            )
        }
        self.bm25 = BM25([w["text"] for w in self.windows])
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        import numpy as np

        # Stage 1: candidate union
        qvec = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ qvec
        semantic = set(np.argsort(-sims)[:CANDIDATES_PER_STAGE].tolist())
        lexical = {i for i, _ in self.bm25.top(query, CANDIDATES_PER_STAGE)}
        candidates = sorted(semantic | lexical)
        if not candidates:
            return []

        # Stage 2: rerank
        pairs = [(query, self.windows[i]["text"]) for i in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for idx, score in ranked:
            w = self.windows[idx]
            rec = self.records[w["clip_id"]]
            results.append({
                "clip_id": w["clip_id"],
                "title": rec["title"],
                "program": rec["program"],
                "date": rec["date"],
                "decade": rec["decade"],
                "excerpt": w["text"],
                "start_s": w["start"],
                "end_s": w["end"],
                "asr_confidence": w["confidence"],
                "rerank_score": round(float(score), 4),
                "matched_by": ("semantic" if idx in semantic else "")
                              + ("+lexical" if idx in lexical and idx in semantic else "")
                              + ("lexical" if idx in lexical and idx not in semantic else ""),
                "tags": rec["tags"],
                "source_url": rec["source_url"],
                "rights": rec["rights"],
                "audio_file": rec["audio_file"],
            })
        return results


def main() -> int:
    query = " ".join(sys.argv[1:]) or "allied invasion of normandy"
    searcher = Searcher()
    print(f'Query: "{query}"\n')
    for i, r in enumerate(searcher.search(query), 1):
        mins, secs = divmod(int(r["start_s"]), 60)
        print(f"{i}. [{r['rerank_score']:+.2f}] {r['program']} {r['date']} @ {mins}:{secs:02d}  ({r['matched_by']})")
        print(f"   {r['title']}  (asr_conf={r['asr_confidence']:.2f})")
        print(f"   \"{r['excerpt'][:160]}\"")
        print(f"   {r['source_url']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
