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
the dataset schema and loader, the evidence corpus tooling, an evidence verifier, a first
dataset of 40 verified examples with its version and dev/test split, and the first
statistical methods: a paired bootstrap with a measured coverage study, McNemar's exact
test, Holm correction, and an exact power analysis. Everything under "Planned" is not built
yet.

## Current

- Python 3.12 package managed with uv (src layout)
- Dataset example schema with Pydantic validation (see "Dataset format" below)
- JSONL dataset loader that validates every line and reports all problems at once, each
  with its line number
- Content-hash dataset version that changes only when example content changes
- Stable, group-aware dev/test split (40% of groups held out for test), with a check that
  examples sharing evidence share a group
- Evidence corpus tooling: extracts readable text from saved HTML pages, normalizes it for
  exact quote matching, and records a SHA-256 manifest (see "Evidence corpus" below)
- Evidence verifier that checks every quote in a dataset against its corpus document and
  shows where a failing quote stops matching
- A first dataset of 40 hand-written examples on 14 public UCCS pages, with every evidence
  quote verified (see "Dataset" below)
- Paired bootstrap confidence intervals for the difference between two systems, and a
  coverage study that measures how often those intervals are actually right (see
  "Statistics" below)
- McNemar's exact test for paired pass/fail results, and Holm correction for comparing
  many slices at once
- Exact power analysis for McNemar's test: the smallest pass-rate drop a dataset of a given
  size can reliably detect
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

- Grow the dataset to 250+ examples, with more multi-passage, conditional, and
  false-premise questions (the power analysis below shows why 40 is not enough)
- Noise baseline (same config, repeated runs), which also measures the real disagreement
  rate the power analysis currently assumes
- Deterministic scorers plus an LLM judge calibrated against human labels (Cohen's kappa)
- Output caching keyed by config, model, prompt, and dataset version
- CLI: `trustbench run` and `trustbench compare`
- GitHub Action that comments on a PR and fails only on significant regressions
- Systems under test: [InferX](https://github.com/Pacosalazar8403/inferx) routing policies
  first, then a Policy Assistant RAG system built on the same documents

## Statistics

### Paired bootstrap

`trustbench.stats.paired_bootstrap(baseline, candidate)` takes the two systems' scores on
the same examples, in the same order, and returns the mean difference (candidate minus
baseline) with a percentile bootstrap confidence interval, for example
`+0.050 [-0.010, +0.110] (95% CI)`.

- **Paired:** each resample draws whole examples with replacement, keeping both systems'
  scores together. A hard question tends to be hard for both systems, so pairing cancels
  that shared difficulty and the interval measures only how the systems differ.
- **Reproducible:** a seed fixes the resamples, so the same data always gives the same
  interval.
- **Bounded memory:** resamples are drawn in chunks of 1,000.

### Measured coverage

A "95% interval" is only honest if it contains the true difference about 95% of the time.
`python -m trustbench.coverage` simulates 1,000 experiments per size from a model where
the candidate is truly better by 0.03, with the two systems agreeing on most examples as
real systems do, and counts how often the interval contains 0.03:

| Examples | Coverage of a nominal 95% interval |
|----------|------------------------------------|
| 13       | 54.8%                              |
| 40       | 90.5%                              |
| 100      | 94.7%                              |
| 250      | 94.5%                              |

At 100 examples and up, the intervals are about as reliable as they claim. At 13 examples,
the size of the current test split, a "95% interval" contains the truth only about half
the time: with so few examples, many experiments have no example where the systems
disagree at all, so the bootstrap sees no variation and reports a zero-width interval.
**Until the test split is much larger, a bootstrap interval on it must not be read as a
95% interval.** The tests pin both results, so this warning stays backed by code.

### McNemar's exact test

`trustbench.significance.mcnemar_exact(baseline, candidate)` takes the two systems'
pass/fail results on the same examples and returns the number of **wins** (examples only
the candidate passes), **losses** (examples only the baseline passes), and a two-sided
p-value, for example `wins 9, losses 1, p = 0.0215`.

- **Only disagreements count.** Examples both systems pass, or both fail, say nothing about
  which system is better. If the systems were equally good, each disagreement would be a
  win or a loss with equal chance, and the p-value is the chance of a split at least as
  lopsided as the observed one.
- **Exact at any size.** The p-value is computed from whole-number binomial coefficients,
  not from the chi-square approximation that most textbooks show, which is unreliable when
  disagreements are few. A test checks that when the systems are truly equal, the test
  raises a false alarm in at most 5% of simulated experiments.
- **The cost of being exact on small data is power.** A two-sided test at the 5% level needs
  at least **6 disagreements, all in the same direction**, before it can report a
  significant difference (5 give p = 0.0625, 6 give p = 0.031). The tests pin this
  boundary.

### Holm correction for slices

Comparing two systems separately on each category multiplies the chances of a false alarm:
six tests at the 5% level have about a 26% chance that at least one looks significant by
luck. `trustbench.significance.holm_adjust(p_values)` returns adjusted p-values in the same
order as the input, and rejecting every slice whose adjusted p-value is at most 0.05 keeps
the chance of **any** false alarm across all slices at 5%, however the slices are related.
Holm gives the same guarantee as the simpler Bonferroni correction while never rejecting
less often.

### Power analysis

A gate that finds no significant regression has only proved something if it could have
found one. **Power** is the chance that McNemar's test reports a difference when a real one
exists, and the **minimum detectable drop** is the smallest real fall in pass rate that it
catches with 80% power. `python -m trustbench.power` computes it exactly, by summing over
every possible number of disagreements and every split of them, with no simulation; a test
checks the exact result against simulation anyway, and another checks that its rejection
rule agrees with `mcnemar_exact` on every split up to 30 disagreements.

Power depends on how often the two systems disagree, which is an **assumption** until real
systems are run (`--disagreement-rate` changes it). Minimum detectable drop, in percentage
points of pass rate, at 80% power and a two-sided 5% level:

| Examples | 20% disagreement | 10% disagreement |
|----------|------------------|------------------|
| 13       | not detectable   | not detectable   |
| 40       | 19.6             | not detectable   |
| 100      | 12.8             | 8.9              |
| 250      | 8.2              | 5.8              |
| 500      | 5.8              | 4.1              |
| 1000     | 4.1              | 2.9              |

- **The current test split cannot detect any regression.** With 13 examples, even a
  candidate that loses every disagreement is caught less than 80% of the time.
- **Fewer disagreements hurt small datasets and help large ones.** Rare disagreements often
  fall short of the 6 that McNemar's test needs, but when there are enough examples, a
  real drop between similar systems shows up as a very lopsided split.
- **Catching a 5-point drop takes roughly 500 to 1,000 examples**, which sets the goal for
  growing the dataset.

### Which method answers which question

| Question                                              | Method                       |
|-------------------------------------------------------|------------------------------|
| How big is the difference, and how sure are we?       | Paired bootstrap interval    |
| Did the pass rate change, even on a small dataset?    | McNemar's exact test         |
| Which slices changed, with many slices compared?      | McNemar per slice, then Holm |
| Could this dataset have caught a regression at all?   | Power analysis               |

## Dataset

`data/uccs.jsonl` holds 40 questions about UCCS registration, deadlines, billing, housing,
advising, and military benefits. Every question, reference answer, and evidence quote was
written by hand by the author, a UCCS student, and every quote was verified against the
corpus snapshot with the evidence verifier.

To see the dataset version, the split, and any evidence conflicts:

```bash
uv run python -m trustbench.splits data/uccs.jsonl
```

| Category        | Dev | Test |
|-----------------|-----|------|
| `lookup`        | 18  | 8    |
| `unanswerable`  | 3   | 3    |
| `paraphrase`    | 1   | 2    |
| `conditional`   | 2   | 0    |
| `false_premise` | 2   | 0    |
| `multi_passage` | 1   | 0    |
| **Total**       | 27  | 13   |

Difficulty: 22 easy, 16 medium, 2 hard.

### Version and split

- **Version:** a 16-character content hash. Examples are sorted by ID and serialized
  canonically, so line order, key order, and whitespace in the file do not change the
  version; any change to any example does. Results will be tagged with this version.
- **Split:** each group (an example's `group`, or its own ID) is hashed with a fixed salt,
  and about 40% of groups go to **test**. A group's split depends only on its own name, so
  adding or removing other examples never moves an existing one. Dev is for building and
  tuning systems; test is held out for final claims.
- **Groups:** examples that quote the same sentence must share a group, so a near-copy of a
  test question can never sit in dev. CI enforces this. Before groups were added, two
  examples quoting the same census-date sentence were split across dev and test.

### Known limitations

These are stated so that results are not over-read:

- **Too small to detect regressions.** With 40 examples (13 in test), the power analysis
  shows that no regression on the test split can be reliably detected, and bootstrap
  intervals on it are far less reliable than their stated level (see "Statistics" above).
- **Unbalanced test split.** The test split has no conditional, false-premise, or
  multi-passage examples yet, so no held-out claim can be made about those categories. The
  split favors stability over balance (stratifying would move examples whenever new ones
  are added), so balance has to come from adding more of those examples.
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
- `group` is optional and uses the same format as IDs. Examples with the same group always
  land in the same split.
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
`sources.txt`, `sources.txt` and the manifest list the same documents, and examples that
share evidence share a group.

## Development

```bash
uv sync --locked
uv run pre-commit install
```

After `pre-commit install`, every commit runs the hooks automatically. The hooks do not run
the tests, so run them before committing. To run the same checks CI runs:

```bash
uv run pre-commit run --all-files
uv run pytest
```
