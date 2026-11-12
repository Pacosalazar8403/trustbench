"""Blind human labels for system answers, to check an LLM judge against a person.

A judge is trusted only as far as it agrees with people, so a person grades a sample of
answers first, and the judge is compared with those labels. The labels must be **blind**:
the labeler never sees which model wrote an answer or what the judge or the scorers said,
because knowing a verdict pulls a labeler toward it. Answers are shown in a random order,
fixed by a seed, so a session can stop and resume where it left off.

The labeler uses the same rubric as the judge (printed at the start of every session).
If the two graded different things, their disagreement would measure the rubrics, not
the judge.

Most of the sample (75% by default) comes from answers the deterministic scorers cannot
judge, because that is where the judge will be used; the rest come from answers they can,
so human, scorer, and judge can all be compared on the same items.

Labels are saved one per line as soon as they are given, with the answer text, so the
label file stands on its own and is committed with the dataset.

A held-out label set, for confirming a change that was tuned on earlier labels, must not
share questions with them: `--exclude-labels` leaves out every answer to any question
labeled in the given files, not only the same answers, because a prompt tuned on one
model's answer to a question has already been shaped by that question.

Run `python -m trustbench.label data/uccs.jsonl corpus/uccs` after running the models.
"""

import argparse
import random
import textwrap
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from trustbench.dataset import DatasetError, dataset_version, load_examples
from trustbench.evidence import CorpusError, load_corpus_text
from trustbench.retrieval import BM25, build_index
from trustbench.runner import (
    DEFAULT_CACHE,
    DEFAULT_K,
    Answer,
    AnswerCache,
    OllamaModel,
    RunnerError,
    cache_key,
    prompt_for,
    select_examples,
)
from trustbench.schema import Example
from trustbench.scoring import score_answer

LABELS_DIR = Path("data/labels")
DEFAULT_LABELS = LABELS_DIR / "uccs-dev.jsonl"
DEFAULT_MODELS = ("llama3.2:1b", "llama3.2:3b")
DEFAULT_COUNT = 60
PROSE_SHARE = 0.75
WIDTH = 96

RUBRIC = """\
Grade each answer against the reference answer, which is correct.
  correct:   it gives the information the reference gives for what the question asks.
             Different wording is fine. Extra information is fine unless it is wrong.
  incorrect: it contradicts the reference, leaves out something the question needs,
             answers a different question, or says the documents do not say when the
             reference answers. If the reference says the documents do not answer, the
             answer is correct only if it also does not invent an answer.
If you are unsure whether an extra claim is true, check the corpus page, or skip."""

Verdict = Literal["correct", "incorrect", "skip"]
CHOICES: dict[str, Verdict] = {"c": "correct", "i": "incorrect", "s": "skip"}


class LabelError(Exception):
    """The label file could not be read."""


class Label(BaseModel):
    """One person's verdict on one answer, with the answer itself."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    answer_key: str
    example_id: str
    model: str
    answer: str
    label: Verdict
    note: str = ""
    dataset_version: str
    labeled_at: str


@dataclass(frozen=True)
class Candidate:
    """An answer that could be labeled."""

    example: Example
    answer: Answer

    @property
    def prose(self) -> bool:
        """True when the deterministic scorers cannot judge this answer."""
        return not score_answer(self.example, self.answer.answer).applies


def read_labels(path: Path) -> list[Label]:
    """Read a label file; a missing file has no labels yet."""
    if not path.exists():
        return []
    labels = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            labels.append(Label.model_validate_json(line))
        except ValidationError as error:
            raise LabelError(
                f"{path}: line {number} is not a valid label: {error.errors()[0]['msg']}"
            ) from error
    return labels


def append_label(path: Path, label: Label) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(label.model_dump_json() + "\n")


def find_candidates(
    examples: Iterable[Example], index: BM25, cache: AnswerCache, model_names: Sequence[str], k: int
) -> list[Candidate]:
    """Every cached answer, from each model, to each example."""
    candidates = []
    for example in examples:
        prompt = prompt_for(example, index, k)
        for name in model_names:
            answer = cache.get(cache_key(name, prompt))
            if answer is not None:
                candidates.append(Candidate(example, answer))
    return candidates


def plan_sample(
    candidates: Sequence[Candidate], count: int, seed: int, prose_share: float = PROSE_SHARE
) -> list[Candidate]:
    """A fixed random sample, mostly answers the scorers cannot judge, in random order."""
    rng = random.Random(seed)
    prose = sorted((c for c in candidates if c.prose), key=lambda c: c.answer.key)
    judged = sorted((c for c in candidates if not c.prose), key=lambda c: c.answer.key)
    rng.shuffle(prose)
    rng.shuffle(judged)
    take_prose = min(len(prose), round(count * prose_share))
    take_judged = min(len(judged), count - take_prose)
    take_prose = min(len(prose), count - take_judged)
    sample = prose[:take_prose] + judged[:take_judged]
    rng.shuffle(sample)
    return sample


def card(position: int, total: int, candidate: Candidate) -> str:
    """What the labeler sees: the question, reference, evidence, and answer. No model name."""
    example = candidate.example

    def block(title: str, text: str) -> str:
        return textwrap.fill(text, WIDTH, initial_indent=f"{title:11}", subsequent_indent=" " * 11)

    lines = [f"---- {position} of {total} " + "-" * 60]
    lines.append(block("QUESTION", example.question))
    lines.append(block("REFERENCE", example.reference_answer))
    for item in example.evidence:
        lines.append(block("EVIDENCE", f"({item.doc_id}) {item.quote}"))
    if not example.evidence:
        lines.append(block("EVIDENCE", "(none: the documents do not answer this question)"))
    lines.append(block("ANSWER", candidate.answer.answer))
    return "\n".join(lines)


def label_session(
    sample: Sequence[Candidate],
    path: Path,
    version: str,
    ask: Callable[[str], str],
    show: Callable[[str], None],
) -> int:
    """Ask for a verdict on each unlabeled answer in the sample; return how many were given."""
    done = {label.answer_key for label in read_labels(path)}
    given = 0
    for position, candidate in enumerate(sample, start=1):
        if candidate.answer.key in done:
            continue
        show(card(position, len(sample), candidate))
        try:
            choice = ""
            while choice not in CHOICES and choice != "q":
                choice = ask("[c]orrect  [i]ncorrect  [s]kip  [q]uit > ").strip().lower()
            if choice == "q":
                break
            note = ask("note (optional) > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        append_label(
            path,
            Label(
                answer_key=candidate.answer.key,
                example_id=candidate.example.id,
                model=candidate.answer.model,
                answer=candidate.answer.answer,
                label=CHOICES[choice],
                note=note,
                dataset_version=version,
                labeled_at=datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        given += 1
    return given


def main(
    argv: Sequence[str] | None = None,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trustbench.label",
        description="Label a fixed random sample of cached answers, blind to model and judge.",
    )
    parser.add_argument("dataset", type=Path, help="JSONL dataset file")
    parser.add_argument("corpus_dir", type=Path, help="corpus folder with a built manifest")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="answers to label")
    parser.add_argument("--seed", type=int, default=0, help="fixes the sample and its order")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="label file")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="chunks retrieved per question")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="answer cache file")
    parser.add_argument(
        "--exclude-labels",
        type=Path,
        nargs="*",
        default=[],
        help="label files whose questions are left out entirely (for a held-out set)",
    )
    args = parser.parse_args(argv)
    if args.count < 1 or args.k < 1:
        parser.error("--count and --k must be at least 1")
    if args.labels in args.exclude_labels:
        parser.error("--labels cannot also be in --exclude-labels")

    try:
        dataset = load_examples(args.dataset)
        examples = select_examples(dataset, args.split)
        index = build_index(load_corpus_text(args.corpus_dir))
        cache = AnswerCache(args.cache)
        before = len(read_labels(args.labels))
        excluded = {label.example_id for path in args.exclude_labels for label in read_labels(path)}
    except (DatasetError, CorpusError, RunnerError, LabelError, FileNotFoundError) as error:
        show(f"error: {error}")
        return 2

    names = [OllamaModel(model).name for model in args.models]
    examples = [example for example in examples if example.id not in excluded]
    candidates = find_candidates(examples, index, cache, names, args.k)
    if not candidates:
        show(f"error: no cached answers from {', '.join(args.models)} on {args.split}")
        return 2
    sample = plan_sample(candidates, args.count, args.seed)
    show(RUBRIC)
    if excluded:
        show(f"\nleaving out {len(excluded)} questions already labeled in other files")
    show(
        f"\n{len(sample)} answers in this sample ({sum(c.prose for c in sample)} that the "
        f"scorers cannot judge), {before} labels so far in {args.labels}\n"
    )
    given = label_session(sample, args.labels, dataset_version(dataset), ask, show)
    labeled = {label.answer_key for label in read_labels(args.labels)}
    remaining = sum(1 for candidate in sample if candidate.answer.key not in labeled)
    show(f"\n{given} labels saved this session; {remaining} of {len(sample)} left")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
