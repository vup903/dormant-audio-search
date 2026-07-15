# In-browser demo

**Live: https://vup903.github.io/dormant-audio-search/**

The same two-stage retrieval as the Python app, ported line-for-line to
JavaScript ([site/retrieval.mjs](site/retrieval.mjs)) and run entirely in the
browser with ONNX models via transformers.js — no server. Corpus embeddings
are computed at build time with the exact quantized model the browser uses
for queries, so the two share one vector space. Audio streams from
archive.org at the result's timecode.

```
npm install          # once
node build_index.mjs # windows + embeddings + models -> dist/
node eval.mjs        # hit@5 on eval/golden.yaml, compared to the Python baseline
```

Deploy: push `dist/` to the `gh-pages` branch.

What differs from the Python app: models are int8-quantized (~50 MB download
instead of a server), and the eval report page isn't included. The retrieval
logic, constants, and result fields are identical, and `eval.mjs` holds the
port to the same hit@5 floor (0.8) as `eval/test_retrieval.py`.
