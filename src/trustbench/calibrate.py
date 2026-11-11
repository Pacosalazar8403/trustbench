"""Calibrate the LLM judge against human labels: how far can its verdicts be trusted?

Every labeled answer is judged (judgments are cached, so answers the judge already graded
cost nothing), and the judge's verdicts are compared with the person's using Cohen's
kappa. The comparison that matters most is on answers the deterministic scorers cannot
judge, because that is where the judge would be used. Agreement on 60 labels is itself
uncertain, so kappa comes with a bootstrap interval.

The judge is trusted for gating only if kappa reaches `TRUST_KAPPA` (0.6, the start of
"substantial" agreement), a common bar for automated graders. The report also shows
whether the judge is more lenient or stricter than the person, and lists every
disagreement with the judge's reason and the person's note.

Run `python -m trustbench.calibrate data/uccs.jsonl` after labeling.
"""

import argparse
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from trustbench.agreement import agreement
from trustbench.dataset import DatasetError, dataset_version, load_examples
from trustbench.judge import DEFAULT_JUDGE, DEFAULT_JUDGMENTS, build_judge_prompt, parse_verdict
from trustbench.label import DEFAULT_LABELS, Label, LabelError, read_labels
from trustbench.runner import (
    OLLAMA_URL,
    AnswerCache,
    Model,
    OllamaModel,
    RunnerError,
    generate_cached,
)
from trustbench.schema import Example
from trustbench.scoring import Score, score_answer

TRUST_KAPPA = 0.6
RESAMPLES = 2_000


@dataclass(frozen=True)
class Item:
    """One labeled answer with all three verdicts: the person's, the judge's, the scorers'."""

    label: Label
    example: Example
    judge: Score
    scorer: Score

    @property
    def human(self) -> bool:
        return self.label.label == "correct"


def kappa_interval(
    first: Sequence[bool], second: Sequence[bool], resamples: int = RESAMPLES, seed: int = 0
) -> tuple[float, float] | None:
    """A 95% percentile bootstrap interval for kappa, resampling items with replacement.

    Resamples where kappa is undefined (every verdict the same) are left out. Returns None
    when too few resamples have a defined kappa to form an interval.
    """
    rng = random.Random(seed)
    count = len(first)
    values = []
    for _ in range(resamples):
        picks = [rng.randrange(count) for _ in range(count)]
        kappa = agreement([first[i] for i in picks], [second[i] for i in picks]).kappa
        if kappa is not None:
            values.append(kappa)
    if len(values) < resamples // 2:
        return None
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def judge_labels(
    labels: Sequence[Label],
    examples: dict[str, Example],
    judge: Model,
    cache: AnswerCache,
    version: str,
) -> list[Item]:
    """Judge every labeled answer that was not skipped, using cached judgments if possible."""
    items = []
    for label in labels:
        if label.label == "skip":
            continue
        example = examples[label.example_id]
        reply, _ = generate_cached(
            judge, build_judge_prompt(example, label.answer), cache, example.id, version
        )
        items.append(
            Item(label, example, parse_verdict(reply.answer), score_answer(example, label.answer))
        )
    return items


def compare_line(name: str, first: Sequence[bool], second: Sequence[bool]) -> str:
    """One row: agreement, kappa, and kappa's interval, or why there is no row."""
    if not first:
        return f"{name}: no items"
    table = agreement(first, second)
    interval = kappa_interval(first, second)
    spread = "" if interval is None else f", 95% CI [{interval[0]:.2f}, {interval[1]:.2f}]"
    return f"{name} ({len(first)} answers): {table}{spread}"


def report(items: Sequence[Item], exclude: set[str]) -> list[str]:
    """Agreement between person, judge, and scorers, and whether the judge is trusted."""
    valid = [item for item in items if item.judge.applies]
    prose = [item for item in valid if not item.scorer.applies]
    scored = [item for item in valid if item.scorer.applies]
    blind = [item for item in valid if item.example.id not in exclude]

    def with_judge(group: Sequence[Item]) -> tuple[list[bool], list[bool]]:
        return [item.human for item in group], [bool(item.judge.passed) for item in group]

    def with_scorers(group: Sequence[Item]) -> tuple[list[bool], list[bool]]:
        return [item.human for item in group], [bool(item.scorer.passed) for item in group]

    lines = [
        f"{len(items)} labeled answers, {len(items) - len(valid)} without a judge verdict",
        compare_line("person vs judge, all", *with_judge(valid)),
        compare_line("person vs judge, scorers cannot judge", *with_judge(prose)),
        compare_line("person vs judge, scorers can judge", *with_judge(scored)),
        compare_line("person vs scorers", *with_scorers(scored)),
    ]
    if exclude:
        seen = len(valid) - len(blind)
        lines.append(compare_line(f"person vs judge, without {seen} seen", *with_judge(blind)))
    if valid:
        human_rate = sum(item.human for item in valid) / len(valid)
        judge_rate = sum(bool(item.judge.passed) for item in valid) / len(valid)
        lines.append(f"pass rate: person {human_rate:.0%}, judge {judge_rate:.0%}")
        table = agreement(*with_judge(valid))
        lines += [
            f"  person pass, judge pass {table.both_pass:>4}   "
            f"person pass, judge fail {table.only_first:>4}",
            f"  person fail, judge pass {table.only_second:>4}   "
            f"person fail, judge fail {table.both_fail:>4}",
        ]
    for item in valid:
        if item.human != item.judge.passed:
            note = f" | note: {item.label.note}" if item.label.note else ""
            lines.append(
                f"  disagree {item.example.id} ({item.label.model.split()[0]}): person "
                f"{'pass' if item.human else 'fail'}, judge "
                f"{'pass' if item.judge.passed else 'fail'}: {item.judge.reason}{note}"
            )
    kappa = agreement(*with_judge(prose)).kappa if prose else None
    if kappa is not None and kappa >= TRUST_KAPPA:
        lines.append(
            f"verdict: TRUSTED (kappa {kappa:.2f} >= {TRUST_KAPPA} on answers the scorers "
            "cannot judge)"
        )
    else:
        shown = "undefined" if kappa is None else f"{kappa:.2f}"
        lines.append(
            f"verdict: NOT TRUSTED (kappa {shown} < {TRUST_KAPPA} on answers the scorers "
            "cannot judge)"
        )
    return lines


def main(
    argv: Sequence[str] | None = None,
    make_judge: Callable[[str, str], Model] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trustbench.calibrate",
        description="Compare the LLM judge with human labels, and decide whether to trust it.",
    )
    parser.add_argument("dataset", type=Path, help="JSONL dataset file")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="label file")
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help="Ollama model that judges")
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="example IDs whose labels were not blind; also reported without them",
    )
    args = parser.parse_args(argv)

    try:
        dataset = load_examples(args.dataset)
        labels = read_labels(args.labels)
        cache = AnswerCache(args.judgments)
    except (DatasetError, LabelError, RunnerError, FileNotFoundError) as error:
        print(f"error: {error}")
        return 2
    examples = {example.id: example for example in dataset}
    unknown = sorted({label.example_id for label in labels} - set(examples))
    if unknown:
        print(f"error: labels refer to examples not in the dataset: {', '.join(unknown)}")
        return 2
    if not labels:
        print(f"error: no labels in {args.labels}")
        return 2

    if make_judge is None:
        judge: Model = OllamaModel(args.judge, args.ollama_url, json_output=True)
    else:
        judge = make_judge(args.judge, args.ollama_url)
    print(f"judge {judge.name}")
    print(f"labels {args.labels}", flush=True)
    try:
        items = judge_labels(labels, examples, judge, cache, dataset_version(dataset))
    except RunnerError as error:
        print(f"error: {error}")
        return 2
    for line in report(items, set(args.exclude)):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
