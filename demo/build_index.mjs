// Build the static in-browser demo into demo/dist/:
//   data/demo-index.json  — windows + record metadata (incl. archive.org file_url)
//   data/embeddings.f32   — one normalized MiniLM vector per window, raw Float32
//   models/               — self-hosted ONNX models (embedder + cross-encoder)
//   *.wasm / *.mjs / transformers.min.js — self-hosted transformers.js runtime
//
// Corpus embeddings are computed HERE with the exact same quantized model
// (Xenova/all-MiniLM-L6-v2, q8) the browser uses for queries, so query and
// corpus share one vector space — no Python/JS drift.
//
// Run: node demo/build_index.mjs   (then deploy demo/dist to GitHub Pages)
import { readFileSync, readdirSync, writeFileSync, mkdirSync, cpSync, rmSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { env, pipeline, AutoTokenizer, AutoModelForSequenceClassification } from '@huggingface/transformers';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const DIST = join(HERE, 'dist');
const CACHE = join(HERE, '.model-cache');
env.cacheDir = CACHE;

const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2';
const RERANK_MODEL = 'Xenova/ms-marco-MiniLM-L-6-v2';
const DIM = 384;
const TARGET_SECONDS = 45.0;

// --- windows: port of das/windows.py ---
function mergeSegments(segments) {
  const total = segments.reduce((a, s) => a + (s.end - s.start), 0) || 1.0;
  const confidence =
    segments.reduce((a, s) => a + (s.end - s.start) * s.confidence, 0) / total;
  return {
    start: segments[0].start,
    end: segments[segments.length - 1].end,
    text: segments.map((s) => s.text).join(' '),
    confidence: Math.round(confidence * 1e4) / 1e4,
  };
}

function buildWindows(segments, targetSeconds = TARGET_SECONDS) {
  const windows = [];
  let current = [];
  for (const seg of segments) {
    if (!seg.text.trim()) continue;
    current.push(seg);
    if (current[current.length - 1].end - current[0].start >= targetSeconds) {
      windows.push(mergeSegments(current));
      current = [];
    }
  }
  if (current.length) windows.push(mergeSegments(current));
  return windows;
}

// --- load records + manifest ---
const recDir = join(ROOT, 'data', 'records');
const records = readdirSync(recDir)
  .filter((f) => f.endsWith('.json'))
  .sort()
  .map((f) => JSON.parse(readFileSync(join(recDir, f), 'utf-8')));
const manifest = JSON.parse(readFileSync(join(ROOT, 'data', 'manifest.json'), 'utf-8'));
const fileUrl = new Map(manifest.map((m) => [m.id, m.file_url]));

const windows = [];
for (const rec of records)
  for (const w of buildWindows(rec.segments))
    windows.push({
      clip_id: rec.id,
      start: w.start,
      end: w.end,
      text: w.text,
      index_text: `${rec.title}. ${rec.program}, ${rec.date}. ${w.text}`,
      confidence: w.confidence,
    });
console.log(`windows: ${windows.length} from ${records.length} records`);

// --- embed corpus with the browser's own model ---
const embedder = await pipeline('feature-extraction', EMBED_MODEL, { dtype: 'q8' });
const embeddings = new Float32Array(windows.length * DIM);
const BATCH = 16;
for (let i = 0; i < windows.length; i += BATCH) {
  const batch = windows.slice(i, i + BATCH).map((w) => w.index_text);
  const out = await embedder(batch, { pooling: 'mean', normalize: true });
  embeddings.set(out.data, i * DIM);
  process.stdout.write(`\rembedded ${Math.min(i + BATCH, windows.length)}/${windows.length}`);
}
console.log();

// pull the reranker into the cache too, same dtype the browser will request
await AutoTokenizer.from_pretrained(RERANK_MODEL);
await AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL, { dtype: 'q8' });

// --- assemble dist ---
const recMeta = {};
for (const rec of records)
  recMeta[rec.id] = {
    title: rec.title,
    program: rec.program,
    date: rec.date,
    decade: rec.decade,
    tags: rec.tags,
    source_url: rec.source_url,
    rights: rec.rights,
    file_url: fileUrl.get(rec.id) || null,
  };

rmSync(DIST, { recursive: true, force: true });
mkdirSync(join(DIST, 'data'), { recursive: true });
writeFileSync(
  join(DIST, 'data', 'demo-index.json'),
  JSON.stringify({ dim: DIM, embed_model: EMBED_MODEL, rerank_model: RERANK_MODEL, windows, records: recMeta }),
);
writeFileSync(join(DIST, 'data', 'embeddings.f32'), Buffer.from(embeddings.buffer));

cpSync(join(HERE, 'site'), DIST, { recursive: true });
cpSync(CACHE, join(DIST, 'models'), { recursive: true });

cpSync(
  join(HERE, 'node_modules', '@huggingface', 'transformers', 'dist', 'transformers.min.js'),
  join(DIST, 'transformers.min.js'),
);
// onnxruntime picks a wasm variant (plain/jsep/asyncify/jspi) per browser —
// ship them all; a visitor downloads only the pair their browser selects.
const ortDist = join(HERE, 'node_modules', 'onnxruntime-web', 'dist');
for (const f of readdirSync(ortDist))
  if (f.startsWith('ort-wasm-simd-threaded')) cpSync(join(ortDist, f), join(DIST, f));
writeFileSync(join(DIST, '.nojekyll'), '');
console.log('dist ready:', DIST);
