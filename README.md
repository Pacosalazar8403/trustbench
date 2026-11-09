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
the dataset schema and loader, the evidence corpus tooling, an evidence verifier, and a
first dataset of 40 verified examples. Everything under "Planned" is not built yet.

## Current

- Python 3.12 package managed with uv (src layout)
- Dataset example schema with Pydantic validation (see "Dataset format" below)
- JSONL dataset loader that validates every line and reports all problems at once, each
  with its line number
- Evidence corpus tooling: extracts readable text from saved HTML pages, normalizes it for
  exact quote matching, and records a SHA-256 manifest (see "Evidence corpus" below)
- Evidence verifier that checks every quote in a dataset against its corpus document and
  shows where a failing quote stops matching
- A first dataset of 40 hand-written examples on 14 public UCCS pages, with every evidence
  quote verified (see "Dataset" below)
- Tests with pytest (strict markers and config), including consistency checks between the
  committed dataset, source list, and manifest
- pre-commit hooks on every commit: file hygiene (final newlines, trailing whitespace,
  valid YAML and TOML, merge-conflict markers, large files) plus ruff lint, ruff format,
  and strict mypy, run through uv so they use the exact versions in `uv.lock`
- CI on every pull request: `lint` (all pre-commit hooks) and `test` (pytest), with all
  actions pinned by commit SHA and read-only permissions
- `main` is protected: changes land only through squash-merged pull requests with passing
  checks

## Planned

- Grow the dataset to 200+ examples, with more multi-passage, conditional, and
  false-premise questions
- A content-hash dataset version and a fixed dev split and held-out test split
- Paired bootstrap confidence intervals, McNemar's test, and Holm correction for slices
- Noise baseline (same config, repeated runs) and power analysis (minimum detectable effect)
- Deterministic scorers plus an LLM judge calibrated against human labels (Cohen's kappa)
- Output caching keyed by config, model, prompt, and dataset version
- CLI: `trustbench run` and `trustbench compare`
- GitHub Action that comments on a PR and fails only on significant regressions
- Systems under test: [InferX](https://github.com/Pacosalazar8403/inferx) routing policies
  first, then a Policy Assistant RAG system built on the same documents

## Dataset

`data/uccs.jsonl` holds 40 questions about UCCS registration, deadlines, billing, housing,
advising, and military benefits. Every question, reference answer, and evidence quote was
written by hand by the author, a UCCS student, and every quote was verified against the
corpus snapshot with the evidence verifier.

| Category        | Examples |
|-----------------|----------|
| `lookup`        | 26       |
| `unanswerable`  | 6        |
| `paraphrase`    | 3        |
| `conditional`   | 2        |
| `false_premise` | 2        |
| `multi_passage` | 1        |

Difficulty: 22 easy, 16 medium, 2 hard.

Known limitations, stated so that results are not over-read:

- **Small.** With 40 examples, only large differences between systems can be detected
  reliably. Any comparison reported on this version must say so.
- **Lookup-heavy.** About two thirds of the examples are direct lookups, which most systems
  answer well. The harder categories are underrepresented.
- **One author.** The labels have not been checked by a second person, so there is no
  inter-annotator agreement measurement yet.
- **Time-sensitive.** Some answers depend on the Spring 2027 calendar and on pages that
  can change. The corpus manifest records exactly which snapshot the dataset was written
  against.
- **Sources disagree.** At least one other UCCS page states the first-year housing age
  requirement differently from the page this dataset cites; that page is left out of the
  corpus until the conflict is handled.

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

## Evidence corpus

Dataset evidence quotes public UCCS web pages. The page text is **not** stored in this
repository: republishing full pages is a different thing from quoting them, so the text
stays on the machine that downloaded it. What is committed in `corpus/uccs/`:

- `sources.txt`: one `doc_id url` per line
- `manifest.jsonl`: for each page, its URL, when it was downloaded (UTC), and a SHA-256
  hash and length of its extracted text

To rebuild the corpus locally:

```bash
cd corpus/uccs
mkdir -p raw
grep -v '^#' sources.txt | while read -r doc url; do
  [ -n "$doc" ] && curl -fsSL --max-time 30 -o "raw/$doc.html" "$url"
done
cd ../..
uv run python -m trustbench.corpus build corpus/uccs
git diff corpus/uccs/manifest.jsonl
```

The rebuilt manifest will always have new download times. A page whose **hash** changed
now shows different text than the snapshot the dataset was written against, so its
evidence quotes need to be checked again.

The hash covers the extracted text, not the raw HTML, because HTML changes constantly
(cache tokens, scripts) even when the words a reader sees do not.

How text is extracted:

- Only the page's `<main>` element is used when present; scripts, styles, navigation, and
  footers are dropped.
- Block elements (paragraphs, headings, list items) each start a new line. Inline tags
  never add spaces, so a word that markup splits, like `C<span>ourse`, stays one word.
- Table rows become `cell | cell` lines.
- Collapsed sections (accordions) are kept, because their text is in the HTML even when a
  browser hides it.

How text is normalized for quote matching: Unicode NFKC (which turns non-breaking spaces
into spaces), curly quotes and long dashes replaced with plain ASCII, and every run of
whitespace collapsed into one space. Letter case and all other punctuation are kept, so a
quote that differs from its page in meaning-bearing ways still fails.

Known limitations: pages that load their content with JavaScript cannot be captured with a
plain download, and pages that are only menus of links are left out of the corpus.

### Verifying evidence

With the corpus built locally:

```bash
uv run python -m trustbench.evidence data/uccs.jsonl corpus/uccs
```

Each document's text is checked against its manifest hash before any quote is checked.
A failing quote is reported with where it stops matching, for example
`the first 38 of 39 characters match; the quote continues '.', the page continues ' What'`.
Exit codes: 0 if every quote is verified, 1 if any quote is not, 2 if the check could not
run (missing files, an invalid dataset, or text that does not match the manifest).

This check runs locally, because the page text is not committed. CI runs the checks that
need only committed files: the dataset loads, every evidence `doc_id` is listed in
`sources.txt`, and `sources.txt` and the manifest list the same documents.

## Development

```bash
uv sync --locked
uv run pre-commit install
```

After `pre-commit install`, every commit runs the hooks automatically. To run the same
checks CI runs:

```bash
uv run pre-commit run --all-files
uv run pytest
```
