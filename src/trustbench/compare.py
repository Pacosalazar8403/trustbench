"""Compare a baseline and a candidate on the same examples, and decide whether it regressed.

`compare` reads both systems' answers from the runner's cache; it never calls a model, so
it is fast and repeatable, and a comparison always uses exactly the answers that were
inspected. Prompts are rebuilt with the runner's own code, so a cached answer is found
only if it was made from the same question, chunks, and prompt template.

Every example that the deterministic scorers can judge becomes a pair of pass/fail
results. From those pairs:

- **McNemar's exact test** on all judged examples decides the verdict. The gate fails only
  on a significant regression: more losses than wins, with a p-value at or below alpha.
- **A paired bootstrap interval** shows how large the change is and how sure we are.
- **Per-category slices** show where the change happened. Their p-values are Holm-adjusted
  across the slices, and they are reported for diagnosis only: adding them to the verdict
  would raise the chance of a false alarm beyond alpha.
- **The power analysis**, at the observed disagreement rate, says how large a drop this
  comparison could reliably have caught, so "no significant change" is not over-read.

Run `python -m trustbench.compare data/uccs.jsonl corpus/uccs --baseline llama3.2:1b
--candidate llama3.2:3b` after running both models on the same split.
"""

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from trustbench.dataset import DatasetError, load_examples
from trustbench.evidence import CorpusError, load_corpus_text
from trustbench.power import minimum_detectable_drop
from trustbench.retrieval import BM25, build_index
from trustbench.runner import (
    DEFAULT_CACHE,
    DEFAULT_K,
    AnswerCache,
    OllamaModel,
    RunnerError,
    cache_key,
    prompt_for,
    select_examples,
)
from trustbench.schema import Example
from trustbench.scoring import score_answer
from trustbench.significance import McNemarResult, holm_adjust, mcnemar_exact
from trustbench.stats import BootstrapResult, paired_bootstrap

DEFAULT_ALPHA = 0.05
# Bootstrap intervals measured well below their stated coverage under this many examples.
SMALL_SAMPLE = 100


class Verdict(StrEnum):
    REGRESSION = "REGRESSION"
    IMPROVEMENT = "IMPROVEMENT"
    NO_SIGNIFICANT_CHANGE = "NO SIGNIFICANT CHANGE"


@dataclass(frozen=True)
class Pair:
    """One judged example: whether the baseline and the candidate passed it."""

    example: Example
    baseline: bool
    candidate: bool


@dataclass(frozen=True)
class Slice:
    """The comparison restricted to one category."""

    name: str
    pairs: tuple[Pair, ...]
    test: McNemarResult
    adjusted_p: float


@dataclass(frozen=True)
class Comparison:
    """Everything `compare_answers` found, and the verdict it leads to."""

    pairs: tuple[Pair, ...]
    not_judged: int
    overall: McNemarResult
    interval: BootstrapResult
    slices: tuple[Slice, ...]
    alpha: float

    @property
    def verdict(self) -> Verdict:
        if self.overall.p_value > self.alpha or self.overall.wins == self.overall.losses:
            return Verdict.NO_SIGNIFICANT_CHANGE
        if self.overall.losses > self.overall.wins:
            return Verdict.REGRESSION
        return Verdict.IMPROVEMENT

    @property
    def disagreement_rate(self) -> float:
        return self.overall.disagreements / len(self.pairs)


def compare_answers(
    examples: Iterable[Example],
    baseline: Mapping[str, str],
    candidate: Mapping[str, str],
    alpha: float = DEFAULT_ALPHA,
) -> Comparison:
    """Pair both systems' verdicts on every judged example and test the difference.

    `baseline` and `candidate` map example IDs to answers, and must contain every example.
    Examples the deterministic scorers cannot judge are counted but not compared.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    pairs = []
    not_judged = 0
    for example in examples:
        missing = [
            name
            for name, answers in (("baseline", baseline), ("candidate", candidate))
            if example.id not in answers
        ]
        if missing:
            raise ValueError(f"no {' or '.join(missing)} answer for {example.id}")
        old = score_answer(example, baseline[example.id])
        new = score_answer(example, candidate[example.id])
        if old.passed is None or new.passed is None:
            not_judged += 1
        else:
            pairs.append(Pair(example, old.passed, new.passed))
    if not pairs:
        raise ValueError("none of the examples can be judged by the deterministic scorers")

    overall = mcnemar_exact([pair.baseline for pair in pairs], [pair.candidate for pair in pairs])
    interval = paired_bootstrap(
        [float(pair.baseline) for pair in pairs], [float(pair.candidate) for pair in pairs]
    )
    by_category: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_category.setdefault(pair.example.category.value, []).append(pair)
    names = sorted(by_category)
    tests = [
        mcnemar_exact(
            [pair.baseline for pair in by_category[name]],
            [pair.candidate for pair in by_category[name]],
        )
        for name in names
    ]
    adjusted = holm_adjust([test.p_value for test in tests])
    slices = tuple(
        Slice(name, tuple(by_category[name]), test, p)
        for name, test, p in zip(names, tests, adjusted, strict=True)
    )
    return Comparison(tuple(pairs), not_judged, overall, interval, slices, alpha)


def load_answers(
    examples: Sequence[Example], index: BM25, cache: AnswerCache, model_name: str, k: int
) -> tuple[dict[str, str], list[str]]:
    """Look up a model's cached answer for each example; return the answers and the misses."""
    answers: dict[str, str] = {}
    missing: list[str] = []
    for example in examples:
        answer = cache.get(cache_key(model_name, prompt_for(example, index, k)))
        if answer is None:
            missing.append(example.id)
        else:
            answers[example.id] = answer.answer
    return answers, missing


def report(comparison: Comparison) -> list[str]:
    """Describe a comparison, ending with its verdict."""
    pairs = comparison.pairs
    count = len(pairs)
    old_rate = sum(pair.baseline for pair in pairs) / count
    new_rate = sum(pair.candidate for pair in pairs) / count
    lines = [
        f"{count} judged examples ({comparison.not_judged} more need a judge)",
        f"pass rate  baseline {old_rate:.1%}  candidate {new_rate:.1%}  "
        f"difference {100 * (new_rate - old_rate):+.1f} points",
        f"bootstrap  {comparison.interval}",
    ]
    if count < SMALL_SAMPLE:
        lines.append(
            f"           (with under {SMALL_SAMPLE} examples, intervals cover less often "
            "than they claim; see the README)"
        )
    lines += [
        f"McNemar    {comparison.overall}",
        "",
        "by category (Holm-adjusted across categories; for diagnosis, not the verdict)",
        f"  {'category':14} {'judged':>6} {'baseline':>9} {'candidate':>9} "
        f"{'wins':>5} {'losses':>6} {'p':>7} {'adj. p':>7}",
    ]
    for part in comparison.slices:
        lines.append(
            f"  {part.name:14} {len(part.pairs):>6} "
            f"{sum(pair.baseline for pair in part.pairs):>9} "
            f"{sum(pair.candidate for pair in part.pairs):>9} "
            f"{part.test.wins:>5} {part.test.losses:>6} "
            f"{part.test.p_value:>7.4f} {part.adjusted_p:>7.4f}"
        )
    lines.append("")
    lines.append(_power_line(comparison))
    lost = [pair.example.id for pair in pairs if pair.baseline and not pair.candidate]
    won = [pair.example.id for pair in pairs if pair.candidate and not pair.baseline]
    lines.append(f"lost ({len(lost)}): {', '.join(lost) or '-'}")
    lines.append(f"won ({len(won)}): {', '.join(won) or '-'}")
    lines.append("")
    relation = "<=" if comparison.overall.p_value <= comparison.alpha else ">"
    lines.append(
        f"verdict: {comparison.verdict} "
        f"(p = {comparison.overall.p_value:.4f} {relation} alpha {comparison.alpha})"
    )
    return lines


def _power_line(comparison: Comparison) -> str:
    count = len(comparison.pairs)
    rate = comparison.disagreement_rate
    if rate == 0:
        return "power: the systems never disagreed, so this run cannot estimate power"
    drop = minimum_detectable_drop(count, rate, alpha=comparison.alpha)
    if drop is None:
        return (
            f"power: at {count} examples and a {rate:.1%} disagreement rate, no drop is "
            "detected 80% of the time"
        )
    return (
        f"power: at {count} examples and a {rate:.1%} disagreement rate, a drop of "
        f"{100 * drop:.1f} points or more is detected 80% of the time"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trustbench.compare",
        description="Compare two models' cached answers and decide whether the candidate "
        "regressed. Exit codes: 0 no significant regression, 1 significant regression, "
        "2 the comparison could not run.",
    )
    parser.add_argument("dataset", type=Path, help="JSONL dataset file")
    parser.add_argument("corpus_dir", type=Path, help="corpus folder with a built manifest")
    parser.add_argument("--baseline", required=True, help="baseline Ollama model name")
    parser.add_argument("--candidate", required=True, help="candidate Ollama model name")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="chunks retrieved per question")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="answer cache file")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args(argv)
    if args.k < 1 or not 0 < args.alpha < 1:
        parser.error("--k must be at least 1 and --alpha between 0 and 1")

    try:
        examples = select_examples(load_examples(args.dataset), args.split)
        index = build_index(load_corpus_text(args.corpus_dir))
        cache = AnswerCache(args.cache)
    except (DatasetError, CorpusError, RunnerError, FileNotFoundError) as error:
        print(f"error: {error}")
        return 2

    answers = {}
    for role, model in (("baseline", args.baseline), ("candidate", args.candidate)):
        name = OllamaModel(model).name
        print(f"{role:9}  {name}")
        answers[role], missing = load_answers(examples, index, cache, name, args.k)
        if missing:
            print(
                f"error: {len(missing)} of {len(examples)} {args.split} examples have no "
                f"cached answer from {model} (first: {missing[0]}). Run: python -m "
                f"trustbench.runner {args.dataset} {args.corpus_dir} --model {model} "
                f"--split {args.split} --k {args.k}"
            )
            return 2
    print(f"{args.split}: {len(examples)} examples, k={args.k}")

    try:
        comparison = compare_answers(
            examples, answers["baseline"], answers["candidate"], alpha=args.alpha
        )
    except ValueError as error:
        print(f"error: {error}")
        return 2
    for line in report(comparison):
        print(line)
    return 1 if comparison.verdict is Verdict.REGRESSION else 0


if __name__ == "__main__":
    raise SystemExit(main())
