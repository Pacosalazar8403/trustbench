# TrustBench

A statistical regression gate for LLM and RAG systems.

When you change an LLM system (a new model, a retrieval setting, a routing policy), the
answers change, but LLM outputs also vary from run to run on their own. TrustBench answers
one question about a change: **did the system actually get better or worse, or is the
difference just noise?** It runs a baseline and a candidate on the same examples, compares
them with paired statistics, and fails a pull request only when quality regresses
significantly.

## Status

Pre-alpha. This repository currently contains the project skeleton, its quality tooling,
the schema for dataset examples, and a loader for dataset files. Everything under
"Planned" is not built yet.

## Current

- Python 3.12 package managed with uv (src layout)
- Dataset example schema with Pydantic validation (see "Dataset format" below)
- JSONL dataset loader that validates every line and reports all problems at once, each
  with its line number
- Tests with pytest (strict markers and config)
- Linting and formatting with ruff, strict type checking with mypy (pydantic plugin)
- pre-commit hooks for file hygiene: final newlines, trailing whitespace, valid YAML and
  TOML, merge-conflict markers, and large files
- CI on every pull request: `lint` (pre-commit, ruff, mypy) and `test` (pytest), with all
  actions pinned by commit SHA and read-only permissions
- `main` is protected: changes land only through squash-merged pull requests with passing
  checks

## Planned

- Versioned, hand-verified dataset built on public UCCS policy documents, with a fixed dev
  split and a held-out test split
- Corpus snapshot of the source pages, with automatic checks that every evidence quote
  appears in its document
- Paired bootstrap confidence intervals, McNemar's test, and Holm correction for slices
- Noise baseline (same config, repeated runs) and power analysis (minimum detectable effect)
- Deterministic scorers plus an LLM judge calibrated against human labels (Cohen's kappa)
- Output caching keyed by config, model, prompt, and dataset version
- CLI: `trustbench run` and `trustbench compare`
- GitHub Action that comments on a PR and fails only on significant regressions
- Systems under test: [InferX](https://github.com/Pacosalazar8403/inferx) routing policies
  first, then a Policy Assistant RAG system built on the same documents

## Dataset format

A dataset is a JSONL file: one JSON object per line, one example per object. Each example
is one question with a reference answer and the evidence that proves it:

```json
{"id": "withdraw-001", "question": "Can I drop a class in week 3 and get a refund?", "reference_answer": "Yes. Classes dropped before the census date get a full refund.", "category": "lookup", "difficulty": "easy", "evidence": [{"doc_id": "registration-policy", "quote": "Courses dropped before the census date receive a full refund."}]}
```

| Category        | What it tests                                                      |
|-----------------|--------------------------------------------------------------------|
| `lookup`        | The answer is stated directly in one passage                       |
| `paraphrase`    | The question uses different words than the document                |
| `multi_passage` | The answer needs two or more passages (at least two evidence items) |
| `conditional`   | The answer depends on a condition, a deadline, or a calculation    |
| `false_premise` | The question assumes something the policy contradicts              |
| `unanswerable`  | The documents do not contain the answer (no evidence allowed)      |

Rules enforced when an example is loaded:

- Unknown fields are rejected, so a typo like `refrence_answer` fails loudly.
- Text fields cannot be blank. Surrounding whitespace is removed.
- IDs are lowercase words separated by single hyphens, like `withdraw-001`, and must be
  unique within a file.
- Every category except `unanswerable` needs at least one piece of evidence.
- Evidence is a document ID plus an exact quote, not a chunk ID, so it stays valid no
  matter how a RAG system splits documents into chunks.
- Examples are immutable after loading.

Rules for the file itself:

- Blank lines are ignored, and both Unix and Windows line endings are accepted.
- The file must be UTF-8 and contain at least one example.
- Every problem in the file is reported together, with its line number, for example
  `line 7: evidence.0.quote: String should have at least 1 character`.

## Development

```bash
uv sync --locked
uv run pre-commit install
```

Run the same checks CI runs:

```bash
uv run pre-commit run --all-files
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```
