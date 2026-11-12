"""An LLM judge for answers that plain code cannot grade, and a check of how far to trust it.

The deterministic scorers only judge unanswerable questions and answers with checkable
facts. Most reference answers are prose ("Contact your academic advisor"), and only a
reader can tell whether an answer means the same thing. The judge is a separate model,
from a different family than the systems it grades (models tend to prefer their own
outputs), given the question, the reference answer, the evidence behind it, and the
answer to grade. It replies in JSON with a one-sentence reason and then a verdict; asking
for the reason first makes the model reason before it commits to a verdict.

A judge is itself a model and can be wrong, so it is measured before it is trusted. The
first check is free: on the examples the deterministic scorers can judge, the judge's
verdicts are compared with theirs, and Cohen's kappa says how well they agree beyond
chance. The real check is against human labels.

Judgments are cached like answers, keyed by the judge's name (with its settings) and the
full judge prompt, which contains the answer being graded.

Every prompt version that was measured is kept, unchanged, so its cached judgments stay
valid and its results can be reproduced:

- `v1`: the first prompt. Calibrated against human labels, it was too lenient: it passed
  partial answers and "the documents don't say" when the reference answers.
- `v2`: stricter, aimed at those failures. On the same labels it overcorrected: it
  required every detail in the reference, and treated claims the reference does not
  mention as invented.
- `v3`: requires only what the question asks for, and calls an extra claim wrong only if
  it contradicts the reference answer or the evidence.

The default stays `v1` until a newer version beats it on labels it was not designed on.

Run `python -m trustbench.judge data/uccs.jsonl corpus/uccs --model llama3.2:3b` after
running that model on the same split.
"""

import argparse
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from trustbench.agreement import agreement
from trustbench.compare import load_answers
from trustbench.dataset import DatasetError, dataset_version, load_examples
from trustbench.evidence import CorpusError, load_corpus_text
from trustbench.retrieval import build_index
from trustbench.runner import (
    DEFAULT_CACHE,
    DEFAULT_K,
    OLLAMA_URL,
    AnswerCache,
    Model,
    OllamaModel,
    RunnerError,
    generate_cached,
    select_examples,
)
from trustbench.schema import Example
from trustbench.scoring import Score, score_answer

DEFAULT_JUDGE = "qwen2.5:7b"
DEFAULT_PROMPT = "v1"
DEFAULT_JUDGMENTS = Path("runs/judgments.jsonl")

JUDGE_TEMPLATE_V1 = (
    "You are grading an answer to a student's question about university policy. The "
    "reference answer is correct and is based on the evidence below.\n"
    "\n"
    "The answer is correct if it gives the information the reference answer gives for "
    "what the question asks. Different wording is fine. Extra information is fine unless "
    "it contradicts the reference answer or the evidence.\n"
    "The answer is incorrect if it contradicts the reference answer, leaves out something "
    "the question needs, answers a different question, or says the documents do not "
    "contain the answer when the reference answer gives one. If the reference answer says "
    "the documents do not answer the question, the answer is correct only if it also says "
    "so instead of answering.\n"
    "\n"
    "Question: {question}\n"
    "Reference answer: {reference}\n"
    "Evidence:\n"
    "{evidence}\n"
    "\n"
    "Answer to grade: {answer}\n"
    "\n"
    'Reply with JSON only, in this order: {{"reason": "<one sentence>", "verdict": '
    '"correct" or "incorrect"}}'
)

JUDGE_TEMPLATE_V2 = (
    "You are grading an answer to a student's question about university policy. The "
    "reference answer is correct and is based on the evidence below.\n"
    "\n"
    "Grade by these rules:\n"
    "1. Find every point the question needs, using the reference answer. If any of those "
    "points is missing from the answer or wrong in it, the answer is incorrect, even if "
    "the rest is right.\n"
    "2. If the answer says the documents do not contain the answer, or that it does not "
    "know, but the reference answer gives an answer, the answer is incorrect.\n"
    "3. If the reference answer says the documents do not answer the question, the answer "
    "is correct if it does not invent an answer. Saying the documents do not say, or "
    "pointing the student to an office, is correct. Making up steps, rules, or facts is "
    "incorrect.\n"
    "4. Extra information that the reference answer does not mention is not a reason to "
    "mark the answer incorrect, unless it contradicts the reference answer or the "
    "evidence.\n"
    "\n"
    "Question: {question}\n"
    "Reference answer: {reference}\n"
    "Evidence:\n"
    "{evidence}\n"
    "\n"
    "Answer to grade: {answer}\n"
    "\n"
    "In your reason, name each point the question needs and say whether the answer gives "
    "it correctly. If your reason names any point that is missing, wrong, or invented, "
    'the verdict must be "incorrect".\n'
    'Reply with JSON only, in this order: {{"reason": "<one or two sentences>", '
    '"verdict": "correct" or "incorrect"}}'
)

JUDGE_TEMPLATE_V3 = (
    "You are grading an answer to a student's question about university policy. The "
    "reference answer is correct and is based on the evidence below.\n"
    "\n"
    "Grade by these rules:\n"
    "1. Find only the points the question asks for. The reference answer may contain more "
    "than the question needs; a detail the question does not ask about is not required. "
    'For example, if the question is whether an office is open on Saturdays, "No" '
    "answers it even if the reference also lists the weekday hours. If a point the "
    "question asks for is missing from the answer or wrong in it, the answer is "
    "incorrect.\n"
    "2. If the answer says the documents do not contain the answer, or that it does not "
    "know, but the reference answer gives an answer, the answer is incorrect.\n"
    "3. If the reference answer says the documents do not answer the question, the answer "
    "is correct if it does not invent an answer. Saying the documents do not say, or "
    "pointing the student to an office, is correct. Only in this case, making up steps, "
    "rules, or facts is incorrect.\n"
    "4. A claim is not wrong just because the reference answer does not mention it. Call "
    "a claim wrong only if it contradicts the reference answer or the evidence.\n"
    "\n"
    "Question: {question}\n"
    "Reference answer: {reference}\n"
    "Evidence:\n"
    "{evidence}\n"
    "\n"
    "Answer to grade: {answer}\n"
    "\n"
    "In your reason, name the points the question asks for and say whether the answer "
    "gives each one correctly. If your reason names a point the question asks for that is "
    "missing or wrong, or a claim that contradicts the reference answer or the evidence, "
    'the verdict must be "incorrect". Otherwise it must be "correct".\n'
    'Reply with JSON only, in this order: {{"reason": "<one or two sentences>", '
    '"verdict": "correct" or "incorrect"}}'
)

JUDGE_TEMPLATES = {"v1": JUDGE_TEMPLATE_V1, "v2": JUDGE_TEMPLATE_V2, "v3": JUDGE_TEMPLATE_V3}


@dataclass(frozen=True)
class Judgment:
    """The judge's verdict on one example's answer, and the deterministic scorers' verdict."""

    example: Example
    answer: str
    judge: Score
    scorer: Score
    cached: bool


def build_judge_prompt(example: Example, answer: str, prompt: str = DEFAULT_PROMPT) -> str:
    """The question, reference answer, evidence, and the answer to grade, in one prompt version."""
    if prompt not in JUDGE_TEMPLATES:
        raise ValueError(f"unknown judge prompt {prompt!r}; choose from {sorted(JUDGE_TEMPLATES)}")
    evidence = "\n".join(f"- ({item.doc_id}) {item.quote}" for item in example.evidence)
    return JUDGE_TEMPLATES[prompt].format(
        question=example.question,
        reference=example.reference_answer,
        evidence=evidence or "- (none: the documents do not answer this question)",
        answer=answer,
    )


def parse_verdict(reply: str) -> Score:
    """Read the judge's JSON reply; a reply without a clear verdict gives no verdict."""
    try:
        data = json.loads(reply)
    except json.JSONDecodeError:
        return Score(None, f"judge reply is not JSON: {reply[:80]!r}")
    if not isinstance(data, dict):
        return Score(None, f"judge reply is not a JSON object: {reply[:80]!r}")
    verdict = str(data.get("verdict", "")).strip().lower()
    reason = str(data.get("reason", "")).strip() or "no reason given"
    if verdict == "correct":
        return Score(True, reason)
    if verdict == "incorrect":
        return Score(False, reason)
    return Score(None, f"judge gave no verdict: {reply[:80]!r}")


def judge_answers(
    examples: Iterable[Example],
    answers: Mapping[str, str],
    judge: Model,
    cache: AnswerCache,
    version: str,
    on_judgment: Callable[[int, Judgment], None] | None = None,
    prompt: str = DEFAULT_PROMPT,
) -> list[Judgment]:
    """Judge each example's answer, using cached judgments where possible."""
    judgments = []
    for position, example in enumerate(examples, start=1):
        answer = answers[example.id]
        reply, cached = generate_cached(
            judge, build_judge_prompt(example, answer, prompt), cache, example.id, version
        )
        judgment = Judgment(
            example=example,
            answer=answer,
            judge=parse_verdict(reply.answer),
            scorer=score_answer(example, answer),
            cached=cached,
        )
        judgments.append(judgment)
        if on_judgment is not None:
            on_judgment(position, judgment)
    return judgments


def summarize(judgments: Sequence[Judgment]) -> list[str]:
    """How often the judge passed answers, and how well it agrees with the scorers."""
    valid = [judgment for judgment in judgments if judgment.judge.applies]
    lines = [f"{len(judgments)} judgments, {len(judgments) - len(valid)} without a valid verdict"]
    both = [judgment for judgment in valid if judgment.scorer.applies]
    if both:
        table = agreement(
            [bool(judgment.scorer.passed) for judgment in both],
            [bool(judgment.judge.passed) for judgment in both],
        )
        lines += [
            f"judge vs deterministic scorers on {len(both)} examples both can judge: {table}",
            f"  scorer pass, judge pass {table.both_pass:>4}   "
            f"scorer pass, judge fail {table.only_first:>4}",
            f"  scorer fail, judge pass {table.only_second:>4}   "
            f"scorer fail, judge fail {table.both_fail:>4}",
        ]
    prose = [judgment for judgment in valid if not judgment.scorer.applies]
    if prose:
        passed = sum(1 for judgment in prose if judgment.judge.passed)
        lines.append(
            f"judge passed {passed} of {len(prose)} answers the scorers cannot judge "
            f"({passed / len(prose):.0%})"
        )
    lines.extend(
        f"  disagree {judgment.example.id}: scorer "
        f"{'pass' if judgment.scorer.passed else 'fail'}, judge "
        f"{'pass' if judgment.judge.passed else 'fail'}: {judgment.judge.reason}"
        for judgment in both
        if judgment.scorer.passed != judgment.judge.passed
    )
    lines.extend(
        f"  no verdict {judgment.example.id}: {judgment.judge.reason}"
        for judgment in judgments
        if not judgment.judge.applies
    )
    return lines


def main(
    argv: Sequence[str] | None = None,
    make_judge: Callable[[str, str], Model] = lambda model, url: OllamaModel(
        model, url, json_output=True
    ),
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trustbench.judge",
        description="Judge a model's cached answers with an LLM, and measure how well the "
        "judge agrees with the deterministic scorers.",
    )
    parser.add_argument("dataset", type=Path, help="JSONL dataset file")
    parser.add_argument("corpus_dir", type=Path, help="corpus folder with a built manifest")
    parser.add_argument("--model", required=True, help="the model whose answers are judged")
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help="Ollama model that judges")
    parser.add_argument("--prompt", choices=sorted(JUDGE_TEMPLATES), default=DEFAULT_PROMPT)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--limit", type=int, help="judge only the first N examples")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="chunks retrieved per question")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="answer cache file")
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    args = parser.parse_args(argv)
    if args.k < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--k and --limit must be at least 1")

    try:
        dataset = load_examples(args.dataset)
        examples = select_examples(dataset, args.split, args.limit)
        index = build_index(load_corpus_text(args.corpus_dir))
        answer_cache = AnswerCache(args.cache)
        judgment_cache = AnswerCache(args.judgments)
    except (DatasetError, CorpusError, RunnerError, FileNotFoundError) as error:
        print(f"error: {error}")
        return 2

    system = OllamaModel(args.model).name
    answers, missing = load_answers(examples, index, answer_cache, system, args.k)
    if missing:
        print(
            f"error: {len(missing)} of {len(examples)} {args.split} examples have no cached "
            f"answer from {args.model} (first: {missing[0]}). Run: python -m "
            f"trustbench.runner {args.dataset} {args.corpus_dir} --model {args.model} "
            f"--split {args.split} --k {args.k}"
        )
        return 2

    judge = make_judge(args.judge, args.ollama_url)
    print(f"answers {system}")
    print(f"judge   {judge.name}, prompt {args.prompt}")
    print(f"{len(examples)} {args.split} examples, judgments {args.judgments}", flush=True)

    def report(position: int, judgment: Judgment) -> None:
        verdict = {True: "pass", False: "fail", None: "no verdict"}[judgment.judge.passed]
        status = "cached" if judgment.cached else "new"
        print(
            f"[{position}/{len(examples)}] {judgment.example.id} {verdict} ({status})",
            flush=True,
        )

    started = time.perf_counter()
    try:
        judgments = judge_answers(
            examples, answers, judge, judgment_cache, dataset_version(dataset), report, args.prompt
        )
    except RunnerError as error:
        print(f"error: {error}")
        print("judgments finished before the error are saved in the cache")
        return 2
    except KeyboardInterrupt:
        print(f"stopped; {len(judgment_cache)} judgments are saved in {args.judgments}")
        return 130
    for line in summarize(judgments):
        print(line)
    print(f"done in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
