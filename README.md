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
with measured recall, a model runner with an answer cache, a regression gate that
compares two models' answers, an LLM judge with blind human labels and a calibration that
is not yet conclusive, and the statistical methods behind them: a paired bootstrap with a
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
- Model runner: answers each question with a local model served by Ollama from the top 5
  BM25 chunks, caches every answer under a hash of the model, its settings, and the full
  prompt, and scores the run with the deterministic scorers (see "Runner" below)
- Regression gate: compares two models' cached answers with McNemar's exact test and a
  paired bootstrap, breaks the change down by category with Holm correction, reports
  what the comparison could have detected, and exits 1 only on a significant regression
  (see "Comparing two systems" below)
- LLM judge for answers the scorers cannot grade, a blind labeling tool, 60 human labels,
  and a calibration that compares person, judge, and scorers with Cohen's kappa and its
  bootstrap interval (see "LLM judge" below)
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

- A stricter judge prompt (omissions, abstentions, a verdict that follows its reason),
  confirmed on a new set of held-out labels, before the judge gets a vote in the gate
- Grow the dataset toward 500+ examples, with many more false-premise and multi-passage
  questions (the power analysis below shows what each size can detect)
- Noise baseline (same config, repeated runs), which also measures the real disagreement
  rate the power analysis currently assumes
- Better retrieval, measured against the BM25 baseline: embedding and hybrid search, and
  chunking by sentence or section
- A single `trustbench` command for running, comparing, and checking
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

## Runner

`trustbench.runner` turns the dataset into a run of a real system. For each question it
retrieves the top 5 BM25 chunks, numbers them in a prompt that says to answer only from
them, and sends the prompt to a model. When the chunks do not contain the answer, the
prompt asks for one exact sentence, `The documents don't say.`, and a test checks that
the abstention scorer recognizes it, so the prompt and the scorer cannot drift apart.

```bash
uv run python -m trustbench.runner data/uccs.jsonl corpus/uccs --model llama3.2:3b
```

Options: `--split dev|test|all` (default `dev`), `--limit N`, `--k N` (chunks per
question, default 5), and `--cache PATH` (default `runs/answers.jsonl`, which git
ignores). Models are served locally by [Ollama](https://ollama.com) with temperature 0
and a fixed seed, so a run needs no API key and sends no data anywhere.

### Answer cache

Every answer is appended to the cache as soon as it is made, so a run that is stopped
keeps everything it finished. The cache key hashes exactly what decides the model's
output: the model's name with its settings, and the full prompt, which already contains
the question and the retrieved chunks.

- **Rerunning is free.** A second run of the same configuration on the 158 dev questions
  took 0.1 seconds, every answer from the cache.
- **A stale answer is never reused.** A different model, temperature, seed, prompt
  template, chunking, corpus, or retrieved set changes the prompt or the name, and so
  the key. A larger `k` that retrieves the same chunks gives the same prompt, so it
  correctly reuses the answer.
- **The dataset version is recorded, not keyed.** Each answer stores the version it was
  made with, but editing a reference answer does not discard answers whose prompts did
  not change, because reference answers are never shown to the model.

### First comparison: llama3.2:1b and llama3.2:3b on dev

Both models answered all 158 dev questions. The deterministic scorers can judge 43 of
them (9 unanswerable and 34 with key facts):

| Model       | Judged passed | Unanswerable abstained | Key facts passed | Answerable abstained |
|-------------|---------------|------------------------|------------------|----------------------|
| llama3.2:1b | 12 / 43 (28%) | 2 / 9                  | 10 / 34          | 5 / 149              |
| llama3.2:3b | 24 / 43 (56%) | 7 / 9                  | 17 / 34          | 44 / 149             |

On the same 43 examples, 3b passed 12 that 1b failed and failed none that 1b passed.
McNemar's exact test gives `wins 12, losses 0, p = 0.0005`, so the difference is not
noise (see "Comparing two systems" for the full report). The power analysis agrees: 43
examples can only detect a large gap, and this gap is 28 points.

Crossing abstention with retrieval shows what "better" means here. For each answerable
dev question, the table checks whether an evidence quote was in the 5 chunks the model
saw:

| llama3.2:3b               | Answered | Abstained |
|---------------------------|----------|-----------|
| Answer in the chunks (85) | 68       | 17        |
| Answer not in them (64)   | 37       | 27        |

3b abstains correctly far more often than 1b, which answered 61 of the 64 questions whose
answer it was never shown. But it still answers 37 of those 64 without the evidence, and
it refuses 17 questions whose answer was in front of it. It is less wrong, not well
calibrated, and the two failure rates (unsupported answers and refusals) trade against
each other.

Reading the failures found the same pattern more than once: an answer that cites chunks
which do not support it. Asked about an enrollment deposit, 1b answered "According to
documents [1] and [3], you do not have to pay an enrollment deposit", citing chunks about
direct deposit of refunds. Asked whether international students can get UCCS
scholarships, 3b joined "international" from one chunk and "apply for scholarships
through their student portal" from another, and credited the result to the first. The
citations make an invented answer look checked.

### Known limitations

- **The judged examples are not a random sample.** They are the unanswerable questions
  and the ones whose reference has numbers, so the comparison above holds for factual
  lookups and abstention, not for the 115 prose answers that still need a judge.
- **One run per model, on dev.** Repeated runs (the noise baseline) and the test split
  come later. Temperature 0 and a fixed seed gave identical answers on repeated calls on
  this machine, but that is not guaranteed across Ollama versions or hardware.
- **Answer quality is scored, citations are not.** A wrong citation is only caught when
  it comes with a wrong answer the scorers can see.
- **Timings are from one laptop.** One 3b answer of 60 tokens took 192 seconds, which
  points to a stall on the machine rather than the model, so totals are not a reliable
  measure of model speed. The cache records seconds and tokens for every answer.

## Comparing two systems

`trustbench.compare` is the regression gate. It reads a baseline's and a candidate's
answers from the runner's cache, pairs them on every example the deterministic scorers
can judge, and decides whether the candidate got worse:

```bash
uv run python -m trustbench.compare data/uccs.jsonl corpus/uccs \
  --baseline llama3.2:1b --candidate llama3.2:3b
```

It never calls a model, so it is fast, repeatable, and always judges exactly the answers
that were run and inspected. Prompts are rebuilt with the runner's own functions, so an
answer is found only if it came from the same question, chunks, and prompt template. If
any answer is missing, it stops and prints the runner command that would make it.

What it reports, and why each part is there:

- **Verdict:** McNemar's exact test on all judged examples. The gate fails only on a
  **significant regression**: more losses than wins, with p at or below alpha (default
  0.05). A drop of one or two examples is noise, and a gate that fails on noise gets
  switched off.
- **Size of the change:** a paired bootstrap interval for the difference in pass rate,
  with a warning when there are fewer than 100 examples (see "Measured coverage").
- **Where it changed:** one row per category, with Holm-adjusted p-values. These are for
  diagnosis and are not part of the verdict, because failing on any of several slices
  would raise the chance of a false alarm above alpha.
- **What it could have caught:** the smallest drop detected with 80% power, at the
  number of judged examples and the disagreement rate this comparison actually saw, so
  "no significant change" is not read as "no change".
- **Which examples changed:** the IDs lost and won, for error analysis.

Exit codes: 0 if there is no significant regression, 1 if there is one, and 2 if the
comparison could not run (missing answers, a missing corpus, or nothing to judge). A CI
job can fail on 1 without ever mistaking a broken setup for a pass.

On dev, with llama3.2:1b as the baseline and llama3.2:3b as the candidate:

```text
43 judged examples (115 more need a judge)
pass rate  baseline 27.9%  candidate 55.8%  difference +27.9 points
bootstrap  +0.279 [+0.140, +0.419] (95% CI)
McNemar    wins 12, losses 0, p = 0.0005

  category       judged  baseline candidate  wins losses       p  adj. p
  conditional         3         0         1     1      0  1.0000  1.0000
  lookup             26         9        13     4      0  0.1250  0.3750
  multi_passage       5         1         3     2      0  0.5000  1.0000
  unanswerable        9         2         7     5      0  0.0625  0.2500

power: at 43 examples and a 27.9% disagreement rate, a drop of 22.6 points or more is
detected 80% of the time
verdict: IMPROVEMENT (p = 0.0005 <= alpha 0.05)
```

Swapping the two models gives `REGRESSION` and exit code 1: the gate catches a downgrade.

The improvement is significant overall, but no single category is, even before Holm
correction. The 12 wins are spread over four categories, so no slice has enough
disagreements on its own: the unanswerable slice went 5 to 0, one short of the 6
one-way disagreements McNemar's test needs at the 5% level. Pooling is what makes the
overall result possible, and it is also why slices cannot gate.

Known limitations:

- **Only the deterministically judged examples count**, so the gate is blind to changes
  in the prose answers until the LLM judge is calibrated (see "LLM judge").
- **Both runs must use the same split and `k`.** A comparison across different retrieval
  settings needs both settings run first.
- **The bootstrap resamples positions.** The seed is fixed and examples are sorted by ID,
  so the interval is reproducible for the same data, but two datasets with the same
  counts in a different order give slightly different intervals.
- **One run per model.** Whether a model's own answers change between runs is not
  measured yet; that is the noise baseline.

## LLM judge

The deterministic scorers grade only 43 of the 158 dev questions; the rest are prose that
only a reader can grade. `trustbench.judge` is that reader: a local `qwen2.5:7b` served by
Ollama, from a different model family than the llama systems it grades (models tend to
prefer their own outputs). It sees the question, the reference answer, the evidence
quotes, and the answer, and replies in JSON with a one-sentence reason first and then a
verdict, `correct` or `incorrect`. A reply without a clear verdict gets no verdict and is
listed, never counted as a pass or a fail.

```bash
uv run python -m trustbench.judge data/uccs.jsonl corpus/uccs --model llama3.2:3b
```

A judge is a model and can be wrong, so it gets no vote in the gate until it is measured.

### First check: the judge against the scorers

On the 43 answers both can grade, the judge agreed with the scorers on 37 (86%, kappa
0.72). Reading the 6 disagreements against the corpus split them 3 to 3, and showed how
each rater fails:

- **The scorers were wrong on 3.** They matched "24/7" in an answer that gave a different
  crisis line (health-014), cannot catch a wrong extra fact (military-001), and recognize
  abstention only by its wording, so a correct referral failed (tuition-020).
- **The judge was wrong on 3.** It passed an answer that its own reason called incomplete
  (grad-015), credited words from the reference to the answer (library-012), and failed a
  true claim because the reference did not mention it (parking-015).

Neither is ground truth, which is why the judge is calibrated against a person.

### Blind human labels

`python -m trustbench.label data/uccs.jsonl corpus/uccs` shows a fixed random sample of
answers one at a time and records `correct`, `incorrect`, or `skip`. The labeler never
sees which model wrote an answer or what the judge or the scorers said. The sample is 45
answers the scorers cannot grade and 15 they can, from both models. Labels are saved as
they are given, with the answer text, in `data/labels/uccs-dev.jsonl`, and CI checks the
file.

The labeling rubric is the standard the judge is meant to meet, and it differs from the
judge's prompt in two places on purpose: an extra claim is wrong only if it is actually
wrong (the person can check the page), and an unanswerable question is answered
correctly by any reply that does not invent an answer, including a referral.

Before comparing with the judge, one label was corrected against the corpus (an adjudication
pass, done before seeing the judge's verdicts so they could not bias it). Four labels were
not blind, because those answers had been inspected with the judge's verdict earlier; the
calibration reports kappa with and without them.

### Calibration

`python -m trustbench.calibrate data/uccs.jsonl` judges every labeled answer and compares
person, judge, and scorers, each kappa with a 95% bootstrap interval:

| Comparison                               | Answers | Agree | Kappa | 95% CI        |
|------------------------------------------|---------|-------|-------|---------------|
| person vs judge, scorers cannot grade    | 45      | 80%   | 0.61  | [0.37, 0.82]  |
| person vs judge, all                     | 60      | 78%   | 0.57  | [0.36, 0.77]  |
| person vs judge, scorers can grade       | 15      | 73%   | 0.44  | [-0.10, 0.86] |
| person vs scorers                        | 15      | 73%   | 0.44  | [-0.11, 0.86] |
| person vs judge, without the 4 not blind | 56      | 79%   | 0.58  | [0.36, 0.78]  |

The judge is trusted only if the whole interval on prose answers is at or above 0.6, the
start of "substantial" agreement; it is not trusted if the whole interval is below, and
the result is inconclusive otherwise. The first version of this rule used the point
estimate and called 0.61 "TRUSTED"; it was changed, because 60 labels cannot tell 0.61
from 0.37. The verdict is:

```text
verdict: INCONCLUSIVE (kappa 0.61, 95% CI [0.37, 0.82] ... contains 0.6; label more answers)
```

What the comparison does show:

- **The judge is more lenient than the person.** It passed 50% of the labeled answers and
  the person 38%. Of 13 disagreements, 10 are answers the person failed and the judge
  passed. Its reasons show partial answers forgiven ("does not explicitly mention signing
  a waiver"), unsupported additions forgiven ("additional methods not supported by the
  evidence"), a "documents don't say" reply passed when the reference answers, and once a
  reason and verdict that point in opposite directions.
- **The scorers are no better.** On the 15 answers they grade, they agree with the person
  exactly as well as the judge does (kappa 0.44).
- **The labels that were not blind did not matter** (0.58 without them, 0.57 with).

Next: a stricter judge prompt aimed at these failures, confirmed on a new set of labels
drawn with a different seed. Tuning the prompt on these 60 labels and measuring it on
the same 60 would only fit it to one person's grading.

Known limitations:

- **One labeler, 60 labels.** There is no second person to measure how much people
  disagree with each other, which bounds how well any judge can do.
- **The judge sees only the reference's evidence quotes**, so it cannot check a true
  claim from elsewhere on the page.
- **Each judgment takes about 1.3 seconds** on a laptop GPU, roughly 3.5 minutes for the
  dev split.

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
- Links placed side by side are joined with no separator, so separate items can read as
  one sentence. A model merged two VA benefit forms this way ("Chapters 30, 33, 1606 &
  1607 Chapter 35"). Fixing it changes extracted text and hashes, so it needs its own
  change and a rerun of the evidence verifier.

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
