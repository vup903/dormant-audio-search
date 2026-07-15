// Hand-built two-stage retrieval — a line-for-line port of das/bm25.py and
// das/retrieval.py to JavaScript, so the in-browser demo runs the SAME
// pipeline as the Python app: semantic top-8 ∪ BM25 top-8 → cross-encoder
// rerank → top-5, best window per clip.
//
// This module is pure JS (no model code). The embedder and reranker are
// injected, so the same file runs in the browser demo and in the Node
// eval harness (demo/eval.mjs) that scores hit@5 against eval/golden.yaml.

export const CANDIDATES_PER_STAGE = 8;
export const TOP_K = 5;

export const tokenize = (text) => text.toLowerCase().match(/[a-z0-9]+/g) || [];

// Okapi BM25, identical constants and formula to das/bm25.py.
export class BM25 {
  constructor(docs, k1 = 1.5, b = 0.75) {
    this.k1 = k1;
    this.b = b;
    this.docTokens = docs.map(tokenize);
    this.docLen = this.docTokens.map((t) => t.length);
    this.avgLen = docs.length
      ? this.docLen.reduce((a, x) => a + x, 0) / docs.length
      : 0;
    this.docTf = this.docTokens.map((toks) => {
      const tf = new Map();
      for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1);
      return tf;
    });
    const df = new Map();
    for (const tf of this.docTf)
      for (const term of tf.keys()) df.set(term, (df.get(term) || 0) + 1);
    const n = docs.length;
    this.idf = new Map(
      [...df].map(([term, d]) => [term, Math.log(1 + (n - d + 0.5) / (d + 0.5))]),
    );
  }

  score(query, i) {
    const tf = this.docTf[i];
    const dl = this.docLen[i] || 1;
    let s = 0;
    for (const term of tokenize(query)) {
      const f = tf.get(term);
      if (!f) continue;
      s +=
        (this.idf.get(term) || 0) *
        ((f * (this.k1 + 1)) /
          (f + this.k1 * (1 - this.b + (this.b * dl) / this.avgLen)));
    }
    return s;
  }

  top(query, k) {
    return this.docTokens
      .map((_, i) => [i, this.score(query, i)])
      .filter(([, s]) => s > 0)
      .sort((a, b) => b[1] - a[1])
      .slice(0, k);
  }
}

// Keep only the best-scoring window per clip, preserving rank order.
export function dedupeByClip(ranked, windows, k) {
  const seen = new Set();
  const out = [];
  for (const [idx, score] of ranked) {
    const clip = windows[idx].clip_id;
    if (seen.has(clip)) continue;
    seen.add(clip);
    out.push([idx, score]);
    if (out.length === k) break;
  }
  return out;
}

// Cosine over normalized vectors = dot product (embeddings: Float32Array n*dim).
export function semanticTop(embeddings, dim, qvec, k) {
  const n = embeddings.length / dim;
  const sims = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const off = i * dim;
    for (let j = 0; j < dim; j++) s += embeddings[off + j] * qvec[j];
    sims[i] = s;
  }
  const order = [...sims.keys()].sort((a, b) => sims[b] - sims[a]);
  return order.slice(0, k);
}

// embed(query) -> Float32Array(dim), normalized.
// rerank(query, texts[]) -> number[] cross-encoder scores.
export function createSearcher({ windows, records, embeddings, dim, embed, rerank }) {
  const bm25 = new BM25(windows.map((w) => w.index_text));

  return async function search(query, k = TOP_K) {
    // Stage 1: candidate union
    const qvec = await embed(query);
    const semantic = new Set(semanticTop(embeddings, dim, qvec, CANDIDATES_PER_STAGE));
    const lexical = new Set(bm25.top(query, CANDIDATES_PER_STAGE).map(([i]) => i));
    const candidates = [...new Set([...semantic, ...lexical])].sort((a, b) => a - b);
    if (!candidates.length) return [];

    // Stage 2: rerank, then keep each clip's best window
    const scores = await rerank(query, candidates.map((i) => windows[i].index_text));
    let ranked = candidates
      .map((c, j) => [c, scores[j]])
      .sort((a, b) => b[1] - a[1]);
    ranked = dedupeByClip(ranked, windows, k);

    return ranked.map(([idx, score]) => {
      const w = windows[idx];
      const rec = records[w.clip_id];
      const inSem = semantic.has(idx);
      const inLex = lexical.has(idx);
      return {
        clip_id: w.clip_id,
        title: rec.title,
        program: rec.program,
        date: rec.date,
        decade: rec.decade,
        excerpt: w.text,
        start_s: w.start,
        end_s: w.end,
        asr_confidence: w.confidence,
        rerank_score: Math.round(score * 1e4) / 1e4,
        matched_by: (inSem ? 'semantic' : '') + (inSem && inLex ? '+lexical' : '') + (!inSem && inLex ? 'lexical' : ''),
        tags: rec.tags,
        source_url: rec.source_url,
        rights: rec.rights,
        file_url: rec.file_url,
      };
    });
  };
}
