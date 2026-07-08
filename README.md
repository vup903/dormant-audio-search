# dormant-audio-search

Surfacing stories in dormant audio: a weekend-scale prototype that makes a small
archive of public-domain historical radio broadcasts (1930s–1950s) searchable by
natural-language query — every result linked back to the original recording at
the exact timecode.

> **Status: work in progress.** Built as a scaled-down proof of a longer
> archive-indexing plan: index a small slice, prove the quality, write down the
> rules, then show how to scale.

## What it does

- Fetches 30–50 public-domain broadcasts from [archive.org](https://archive.org)
  (Old Time Radio news, 1930–1959), recording `source_url` and `rights` for every item.
- Transcribes locally with faster-whisper, keeping segment timestamps and ASR confidence.
- Tags each recording against a ~20-term controlled vocabulary (schema-validated,
  every tag marked `ai_generated: true`).
- Hand-built two-stage retrieval: semantic + lexical candidates → rerank → top 5.
  No RAG framework.
- Web UI: ask a question in plain language, get ranked segments with transcript
  excerpt, program, date, timecode — click to play the original audio from that moment.
- An evaluation harness (golden query→clip set, hit@5, scores broken down per decade)
  that reruns under pytest.

## Ground rules

- **Public domain only.** No NPR audio is used anywhere in this project.
- **The source stays attached.** Every result links to the original recording.
- **Provenance is explicit.** AI-generated fields are labeled as such.
- **AI suggests, a human decides.**

*(Architecture, evaluation results, known limitations, and "what I'd do with a
real archive" will land here as the build progresses.)*
