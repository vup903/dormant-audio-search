// Score the in-browser pipeline against eval/golden.yaml with the same
// hit@5 metric as eval/test_retrieval.py, using the artifacts in demo/dist/
// and the same injected-model searcher the browser runs (demo/site/retrieval.mjs).
// Compares the result with the committed Python baseline (eval/results.json).
//
// Run after build_index.mjs: node demo/eval.mjs
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { env, pipeline, AutoTokenizer, AutoModelForSequenceClassification } from '@huggingface/transformers';
import { createSearcher } from './site/retrieval.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
env.cacheDir = join(HERE, '.model-cache');

// minimal parser for the fixed structure of eval/golden.yaml
function parseGolden(text) {
  const queries = [];
  let cur = null;
  for (const raw of text.split('\n')) {
    const line = raw.replace(/#.*$/, '').trimEnd();
    let m;
    if ((m = line.match(/^\s*-\s*query:\s*"(.+)"\s*$/))) {
      cur = { query: m[1], decade: null, expected: [] };
      queries.push(cur);
    } else if ((m = line.match(/^\s*decade:\s*"(.+)"\s*$/))) {
      cur.decade = m[1];
    } else if ((m = line.match(/^\s*expected:\s*\[\s*"(.+)"\s*\]\s*$/))) {
      cur.expected.push(m[1]);
    } else if (cur && (m = line.match(/^\s{6,}-\s*"(.+)"\s*$/))) {
      cur.expected.push(m[1]);
    }
  }
  if (queries.length !== 15 || queries.some((q) => !q.decade || !q.expected.length))
    throw new Error(`golden.yaml parse failed: got ${queries.length} queries`);
  return queries;
}

const golden = parseGolden(readFileSync(join(ROOT, 'eval', 'golden.yaml'), 'utf-8'));
const index = JSON.parse(readFileSync(join(HERE, 'dist', 'data', 'demo-index.json'), 'utf-8'));
const buf = readFileSync(join(HERE, 'dist', 'data', 'embeddings.f32'));
const embeddings = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);

const embedder = await pipeline('feature-extraction', index.embed_model, { dtype: 'q8' });
const tokenizer = await AutoTokenizer.from_pretrained(index.rerank_model);
const reranker = await AutoModelForSequenceClassification.from_pretrained(index.rerank_model, { dtype: 'q8' });

const search = createSearcher({
  windows: index.windows,
  records: index.records,
  embeddings,
  dim: index.dim,
  embed: async (q) => (await embedder(q, { pooling: 'mean', normalize: true })).data,
  rerank: async (q, texts) => {
    const inputs = tokenizer(new Array(texts.length).fill(q), {
      text_pair: texts, padding: true, truncation: true,
    });
    const { logits } = await reranker(inputs);
    return Array.from(logits.data);
  },
});

const rows = [];
for (const c of golden) {
  const results = await search(c.query, 5);
  const got = results.map((r) => r.clip_id);
  const hit = c.expected.some((clip) => got.includes(clip));
  rows.push({ ...c, got, hit });
  console.log(`${hit ? 'HIT ' : 'MISS'} [${c.decade}] ${c.query}`);
  if (!hit) console.log(`     expected ${c.expected.join(', ')} — got ${got.join(', ')}`);
}

const per = {};
for (const d of [...new Set(rows.map((r) => r.decade))].sort()) {
  const sub = rows.filter((r) => r.decade === d);
  per[d] = (sub.filter((r) => r.hit).length / sub.length).toFixed(3);
}
const overall = rows.filter((r) => r.hit).length / rows.length;
const baseline = JSON.parse(readFileSync(join(ROOT, 'eval', 'results.json'), 'utf-8'));
console.log(`\nJS in-browser pipeline: overall hit@5 = ${overall.toFixed(3)}`, per);
console.log(`Python baseline:        overall hit@5 = ${baseline.overall.hit_at_5}`,
  Object.fromEntries(Object.entries(baseline.per_decade).map(([d, v]) => [d, v.hit_at_5])));
process.exit(overall >= 0.8 ? 0 : 1);
