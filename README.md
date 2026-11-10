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
the dataset schema and loader, the evidence corpus tooling (HTML pages and PDF documents),
an evidence verifier, a dataset of 248 verified examples on 101 public UCCS documents with
its version and dev/test split, deterministic answer scorers, a BM25 retrieval baseline
with measured recall, and the first statistical methods: a paired bootstrap with a
measured coverage study, McNemar's exact test, Holm correction, and an exact power
analysis. Everything under "Planned" is not built yet.

## Current

- Python 3.12 package managed with uv (src layout)
- Dataset example schema with Pydantic validation (see "Dataset format" below)
- JSONL dataset loader that validates every line and reports all problems at once, each
  with its line number
- Content-hash dataset version that changes only when example content changes
- Stable, group-aware dev/test split (40% of groups held out for test), with a check that
  examples sharing evidence share a group
- Evidence corpus tooling: extracts readable text from saved HTML pages and PDF documents
  (detected by content, not file name), normalizes it for exact quote matching, and
  records a SHA-256 manifest (see "Evidence corpus" below)
- Evidence verifier that checks every quote in a dataset against its corpus document and
  shows where a failing quote stops matching
- A dataset of 248 examples on 101 public UCCS documents (96 web pages and 5 PDFs), with
  every evidence quote verified (see "Dataset" below)
- Deterministic scorers: abstention for unanswerable questions, and key facts (numbers,
  money, percentages, phone numbers, emails) for answerable ones, with a check that every
  reference answer passes its own scorer (see "Scoring" below)
- BM25 retrieval baseline over the corpus, with recall@k measured at the document level
  and at the quote level (see "Retrieval" below)
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

- An LLM judge for the 72% of examples the deterministic scorers cannot judge, calibrated
  against human labels (Cohen's kappa)
- Grow the dataset toward 500+ examples, with many more false-premise and multi-passage
  questions (the power analysis below shows what each size can detect)
- Noise baseline (same config, repeated runs), which also measures the real disagreement
  rate the power analysis currently assumes
- Better retrieval, measured against the BM25 baseline: embedding and hybrid search, and
  chunking by sentence or section
- Output caching keyed by config, model, prompt, and dataset version
- CLI: `trustbench run` and `trustbench compare`
- GitHub Action that comments on a PR and fails only on significant regressions
- Systems under test: [InferX](https://github.com/Pacosalazar8403/inferx) routing policies
  first, then a Policy Assistant RAG system built on the same documents

## Dataset

`data/uccs.jsonl` holds 248 questions about UCCS admissions, financial aid, tuition and
billing, registration, degree requirements, graduation, academic records, technology, the
library and tutoring, disability services, health and wellness, parking, housing, advising,
and military benefits. Every answer is grounded in one or more of 101 public UCCS
documents: 96 web pages and 5 PDFs.

How it was made: the questions were written by the author, a UCCS student who works as a
student ambassador and orientation leader, from questions students actually ask, and the
author researched every reference answer and evidence quote. The first 40 examples were
written by hand. The next 208 were reviewed and edited with AI assistance: answers trimmed
to what their quotes support, near-duplicates merged or grouped, and categories and groups
assigned. Every evidence quote was then checked against the corpus snapshot with the
evidence verifier; 26 of the 236 new quotes failed the first check and were corrected
against the page text.

To see the dataset version, the split, and any evidence conflicts:

```bash
uv run python -m trustbench.splits data/uccs.jsonl
```

| Category        | Dev | Test |
|-----------------|-----|------|
| `lookup`        | 105 | 57   |
| `conditional`   | 28  | 18   |
| `unanswerable`  | 9   | 6    |
| `multi_passage` | 9   | 3    |
| `paraphrase`    | 3   | 5    |
| `false_premise` | 4   | 1    |
| **Total**       | 158 | 90   |

Difficulty: 158 easy, 75 medium, 15 hard. The first 40 examples were rated by hand; later
examples are rated by category (lookup is easy; paraphrase, conditional, and unanswerable
are medium; multi-passage and false-premise are hard).

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
  examples quoting the same census-date sentence were split across dev and test. When a
  new example shares evidence with an existing one, the new example joins the existing
  example's group, so the existing example keeps its split.

### Unanswerable questions

An unanswerable label means **no document in the corpus answers the question**, so it can
become wrong when the corpus grows. When this batch added 87 documents, every unanswerable
question was searched for again across the whole corpus. Two of the original 40 had become
answerable (a parking permit refund rule and whom to contact about course prerequisites)
and were relabeled with evidence; one was rewritten into an answerable question; and one
was dropped because the corpus implied an answer without stating it. A keyword search only
finds candidates: words like "defer" and "attendance" matched pages with unrelated
meanings ("deferred payment plan", "cost of attendance"), so each match was judged by
hand. This check has to be repeated whenever documents are added.

### Known limitations

These are stated so that results are not over-read:

- **Small regressions are still hard to see.** The 90-example test split detects drops of
  about 13.5 points at 20% disagreement (see "Power analysis" below). Smaller real
  regressions will usually pass the gate.
- **Lookup-heavy.** 162 of the 248 examples (65%) are direct lookups, which most systems
  answer well.
- **Few false premises.** There are only 5 false-premise examples, and only 1 is in the
  test split, so no held-out claim can be made about them. Multi-passage examples are also
  scarce (12, with 3 in test). The split favors stability over balance (stratifying would
  move examples whenever new ones are added), so balance has to come from adding more of
  these examples.
- **Concentrated sources.** 18 examples cite the financial aid Satisfactory Academic
  Progress page and 13 cite the advising FAQ for current students. A change to either page
  affects many examples at once, and a system that handles those two pages well looks
  better overall than it is.
- **One author, AI-assisted review.** The labels have not been checked by a second person,
  so there is no inter-annotator agreement measurement yet.
- **Time-sensitive.** Many answers depend on the Spring 2027 calendar, the Fall 2026
  tuition, fee, and payment pages, the 2026 registration handbooks, and prices that change.
  The corpus manifest records exactly which snapshot the dataset was written against.
- **Sources disagree.** Some UCCS pages state the same rule differently: the first-year
  housing age requirement (the conflicting page is left out of the corpus), how long a
  financial aid SAP appeal takes to process (one sentence says within two weeks, another
  says to allow at least two to three; the example's answer gives both), and how long a
  new student account takes to create (1 to 2 business days on one page, 2 on another).

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

## Scoring

`trustbench.scoring.score_answer(example, answer)` turns a system's answer into a verdict:
passed, failed, or not applicable. The scorers so far are deterministic: plain code, so
they are free, instant, reproducible, and every failure comes with a reason such as
`missing phone:7192554444`.

- **Abstention (unanswerable examples):** the answer passes if it says the documents do not
  contain the answer ("the documents don't say...", "I couldn't find..."), and fails if it
  answers anyway.
- **Key facts (answerable examples):** every number, money amount, percentage, phone
  number, and email address in the reference answer must appear in the answer. Facts are
  normalized before comparing, so `$50.00` equals `$50`, `(719) 255-3391` equals
  `719-255-3391`, and `004509` equals `4509`. **Extra facts are never penalized**, so an
  answer that adds a crisis line or a second contact still passes.
- **Not applicable:** an answerable example whose reference answer has no checkable facts
  ("Contact your academic advisor") gets no verdict, rather than a guessed one.

`python -m trustbench.scoring data/uccs.jsonl` shows how much of the dataset these scorers
can judge:

| Scorer        | Examples | Share |
|---------------|----------|-------|
| Abstention    | 15       | 6%    |
| Key facts     | 54       | 22%   |
| Needs a judge | 179      | 72%   |

Code alone can judge 28% of the dataset; most answers are prose that only a judge reading
for meaning can grade, which is why an LLM judge is planned next. A test also runs every
reference answer through its own scorer: a reference that fails means the scorer or the
data is wrong. It caught two unanswerable references that the first version of the
abstention patterns missed.

Known limitations:

- An answer that invents an answer and also says the documents do not mention something
  passes the abstention check, because abstention is detected by phrases.
- Because extra facts are allowed, a made-up extra number is not caught.
- Numbers written as words ("two to four weeks") are not extracted, so a correct answer
  that spells out a number the reference writes in digits fails.
- Facts are compared as sets, so the right numbers in the wrong places (swapped phone
  numbers, for example) still pass.

## Retrieval

A RAG system can only answer from what it retrieves, so retrieval is measured on its own,
before any language model is involved. `trustbench.retrieval` splits every corpus document
into windows of 100 words that overlap by 25 (1,290 chunks from the 101 documents) and
ranks them with BM25, implemented in the module: a chunk scores higher when it contains the
question's rare words, contains them more often (with diminishing returns, `k1 = 1.5`),
and is shorter (`b = 0.75`). Ties are broken by document ID and chunk position, so
rankings are reproducible.

The dataset is the answer key: every answerable example names the documents and quotes
that contain its answer, so recall needs no model and no labels. It is measured at two
levels:

- **Document recall@k:** a chunk from a cited document is in the top k.
- **Quote recall@k:** a chunk that contains an evidence quote in full, from any document,
  is in the top k. This is what a language model would actually need to read.

`python -m trustbench.retrieval data/uccs.jsonl corpus/uccs`:

| Split | Level    | Examples | @1    | @3    | @5    | @10   |
|-------|----------|----------|-------|-------|-------|-------|
| dev   | document | 149      | 47.7% | 64.4% | 73.8% | 81.9% |
| dev   | quote    | 149      | 30.2% | 48.3% | 57.0% | 66.4% |
| test  | document | 84       | 39.3% | 58.3% | 65.5% | 76.2% |
| test  | quote    | 84       | 21.4% | 34.5% | 45.2% | 58.3% |

What the numbers say:

- **Document recall overstates what a model gets.** For 24 dev questions, BM25 finds a
  chunk from the right document but not the chunk that holds the answer. Even with the
  top 10 chunks, a model sees the answer text for about two thirds of dev questions and
  58% of test questions, which caps how well any generator built on this retriever can do.
- **Near-duplicate pages are not the cause.** Several UCCS pages repeat each other's text,
  but only 1 dev question had its quote retrieved from a page other than the cited one.
- **Test looks worse than dev, but the gap is within noise.** With 84 test examples, the
  difference in quote recall@10 (about 8 points) is roughly 1.2 standard errors.
- **Length normalization barely matters here.** On dev, every `b` from 0 to 1 changed
  recall@k by at most one question, because fixed-size chunks are all about the same
  length. The standard `b = 0.75` was kept rather than tuned to noise, and any tuning is
  done on dev only, so test stays a fair held-out check.

Known limitations:

- Keyword matching misses paraphrases: a question about an "override" does not match a
  page about "permission numbers". Embedding or hybrid retrieval is the usual fix.
- Fixed word windows ignore sentence and section boundaries, and a long quote that crosses
  a window boundary counts as a quote miss even when both halves are retrieved.
- There is no stemming, so "refund" and "refunds" are different words.

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
the size of the original test split, a "95% interval" contains the truth only about half
the time: with so few examples, many experiments have no example where the systems
disagree at all, so the bootstrap sees no variation and reports a zero-width interval. The
test split now has 90 examples, close to the 100-example row, but coverage has not been
measured at exactly that size, so intervals on it should be read with that in mind. The
tests pin both results above, so these warnings stay backed by code.

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

- **What the current test split can detect.** The original 13-example test split could not
  reliably detect any regression. The current 90-example test split detects a drop of
  about 13.5 points at 20% disagreement, or 9.3 points at 10%
  (`minimum_detectable_drop(90, 0.2)` and `minimum_detectable_drop(90, 0.1)`).
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

## Evidence corpus

Dataset evidence quotes public UCCS web pages and PDF documents. The document text is
**not** stored in this repository: republishing full pages is a different thing from
quoting them, so the text stays on the machine that downloaded it. What is committed in
`corpus/uccs/`:

- `sources.txt`: one `doc_id url` per line, grouped by topic with `#` comment lines
- `manifest.jsonl`: for each document, its URL, when it was downloaded (UTC), and a SHA-256
  hash and length of its extracted text

To build the corpus locally:

```bash
cd corpus/uccs
mkdir -p raw
grep -v '^#' sources.txt | while read -r doc url; do
  if [ -n "$doc" ] && [ ! -f "raw/$doc.html" ]; then
    curl -fsSL --max-time 30 -o "raw/$doc.html" "$url" || echo "FAILED: $doc $url"
  fi
done
cd ../..
uv run python -m trustbench.corpus build corpus/uccs
git diff corpus/uccs/manifest.jsonl
```

The loop only downloads documents that are not saved yet. Downloading a saved document
again would replace the snapshot its examples were verified against: if the page had
changed since, its hash would change and its quotes could break, mixed in with whatever
new work is being added. To deliberately refresh a document, delete its file in `raw/`,
download it again, rebuild, and rerun the evidence verifier.

The loop saves every document as `raw/<doc_id>.html`, including PDFs. That is intended:
the build decides how to read each file from its content (a PDF starts with the bytes
`%PDF-`), not from its name.

A rebuilt document whose **hash** changed now shows different text than the snapshot the
dataset was written against, so its evidence quotes need to be checked again.

The hash covers the extracted text, not the raw file, because HTML changes constantly
(cache tokens, scripts) even when the words a reader sees do not.

How text is extracted from HTML:

- Only the page's `<main>` element is used when present; scripts, styles, navigation, and
  footers are dropped.
- Block elements (paragraphs, headings, list items) each start a new line. Inline tags
  never add spaces, so a word that markup splits, like `C<span>ourse`, stays one word.
- Table rows become `cell | cell` lines.
- Collapsed sections (accordions) are kept, because their text is in the HTML even when a
  browser hides it.

How text is extracted from PDFs:

- Every page is read in order with [pypdf](https://github.com/py-pdf/pypdf) (BSD license),
  one text line per line.
- A sentence that wraps onto the next line still matches a quote, because normalization
  turns the line break into a single space.
- A file that starts with the PDF signature but cannot be read fails the build with an
  error that names the file.

How text is normalized for quote matching: Unicode NFKC (which turns non-breaking spaces
into spaces and typographic ligatures into separate letters), curly quotes and long dashes
replaced with plain ASCII, and every run of whitespace collapsed into one space. Letter
case and all other punctuation are kept, so a quote that differs from its source in
meaning-bearing ways still fails. In particular, single and double quote marks are not
treated as the same: a quote copied with `'` where the page has `"` fails, which caught
several transcription errors in this dataset.

Known limitations:

- Pages that load their content with JavaScript cannot be captured with a plain download,
  and pages that are only menus of links are left out of the corpus.
- Scanned PDFs have no text layer; the build stops with an error saying so rather than
  writing an empty document.
- PDFs with several columns can come out with lines from different columns interleaved.
- A word hyphenated across a PDF line break (`regis-` then `tration`) does not match a
  quote that spells it whole.
- PDF tables are read as plain lines, not converted to `cell | cell` lines like HTML tables.

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

This check runs locally, because the document text is not committed. CI runs the checks
that need only committed files: the dataset loads, every evidence `doc_id` is listed in
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
