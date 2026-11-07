# TrustBench

A statistical regression gate for LLM and RAG systems.

When you change an LLM system (a new model, a retrieval setting, a routing policy), the
answers change, but LLM outputs also vary from run to run on their own. TrustBench answers
one question about a change: **did the system actually get better or worse, or is the
difference just noise?** It runs a baseline and a candidate on the same examples, compares
them with paired statistics, and fails a pull request only when quality regresses
significantly.

## Status

Pre-alpha. This repository currently contains only the project skeleton. Everything under
"Planned" is not built yet.

## Current

- Python 3.12 package managed with uv (src layout)
- Smoke tests with pytest

## Planned

- Versioned, hand-verified dataset with a fixed dev split and a held-out test split
- Paired bootstrap confidence intervals, McNemar's test, and Holm correction for slices
- Noise baseline (same config, repeated runs) and power analysis (minimum detectable effect)
- Deterministic scorers plus an LLM judge calibrated against human labels (Cohen's kappa)
- Output caching keyed by config, model, prompt, and dataset version
- CLI: `trustbench run` and `trustbench compare`
- GitHub Action that comments on a PR and fails only on significant regressions
- Systems under test: [InferX](https://github.com/Pacosalazar8403/inferx) routing policies
  and a Policy Assistant RAG system

## Development

```bash
uv sync --locked
uv run pytest
```