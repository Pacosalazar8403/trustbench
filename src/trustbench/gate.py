"""The regression gate for pull requests: did the proposed system get worse than main's?

`system.toml` names the results file of the current system. A pull request that changes
the system changes that line, so the gate compares main's `system.toml` (the baseline,
what is shipped) with the pull request's (the candidate, what is proposed). It needs only
the repository: committed results files and the dataset, no models and no corpus text.

Before comparing, it checks that both results files are for the current dataset version
and the same split, and that both answer every example in it; otherwise it stops with
exit code 2 instead of gating on the wrong data. Differences in model, k, prompt, or
corpus are reported, not refused: they are the changes a pull request exists to propose.
If the two systems are the same, there is nothing to test.

The verdict and exit codes are those of `trustbench.compare`: 0 for no significant
regression, 1 for a significant regression, 2 if the gate could not run. With
`--markdown`, it also writes a report for a pull request comment.

Run `python -m trustbench.gate data/uccs.jsonl --baseline base.toml --candidate system.toml`.
"""

import argparse
import tomllib
from collections.abc import Sequence
from pathlib import Path

from trustbench.compare import DEFAULT_ALPHA, Comparison, Verdict, compare_answers, report
from trustbench.dataset import DatasetError, dataset_version, load_examples
from trustbench.results import Results, ResultsError, read_results
from trustbench.runner import select_examples

SYSTEM_FILE = Path("system.toml")
MARKER = "<!-- trustbench-gate -->"
TITLES = {
    Verdict.REGRESSION: "❌ REGRESSION",
    Verdict.IMPROVEMENT: "✅ IMPROVEMENT",
    Verdict.NO_SIGNIFICANT_CHANGE: "⚪ NO SIGNIFICANT CHANGE",
}


class GateError(Exception):
    """The gate cannot compare these two systems."""


def read_system(path: Path) -> Path:
    """Return the results file a `system.toml` names."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise GateError(f"{path}: no such system file") from error
    except tomllib.TOMLDecodeError as error:
        raise GateError(f"{path}: not valid TOML: {error}") from error
    results = data.get("system", {}).get("results")
    if not isinstance(results, str) or not results:
        raise GateError(f'{path}: needs a [system] table with results = "results/<file>.json"')
    return Path(results)


def check_pair(baseline: Results, candidate: Results, version: str) -> None:
    """Refuse to compare results that do not describe the same questions."""
    for role, results in (("baseline", baseline), ("candidate", candidate)):
        if results.dataset_version != version:
            raise GateError(
                f"{role} results are stale: made for dataset {results.dataset_version}, "
                f"but the dataset is now {version}; re-run and re-export them"
            )
    if baseline.split != candidate.split:
        raise GateError(
            f"baseline is on {baseline.split} but candidate is on {candidate.split}; "
            "a paired comparison needs the same split"
        )


def changes(baseline: Results, candidate: Results) -> list[tuple[str, str, str]]:
    """Each setting that differs between the two systems: (name, baseline, candidate)."""
    rows = [
        ("model", baseline.model, candidate.model),
        ("k", str(baseline.k), str(candidate.k)),
        ("prompt", baseline.prompt_sha256[:12], candidate.prompt_sha256[:12]),
        ("corpus", baseline.corpus_sha256[:12], candidate.corpus_sha256[:12]),
    ]
    return [row for row in rows if row[1] != row[2]]


def markdown(
    paths: tuple[Path, Path],
    baseline: Results,
    candidate: Results,
    comparison: Comparison | None,
) -> str:
    """A pull request comment: verdict, what changed, and the full report."""
    title = "⚪ NO SYSTEM CHANGE" if comparison is None else TITLES[comparison.verdict]
    lines = [
        f"## TrustBench gate: {title}",
        "",
        f"Baseline (main): `{paths[0]}`  ",
        f"Candidate (this pull request): `{paths[1]}`",
        "",
    ]
    differences = changes(baseline, candidate)
    if differences:
        lines += ["| Setting | Baseline | Candidate |", "|---|---|---|"]
        lines += [f"| {name} | `{old}` | `{new}` |" for name, old, new in differences]
        lines.append("")
    if comparison is None:
        lines.append("The two systems give the same answers, so there is nothing to test.")
    else:
        lines += ["```text", *report(comparison), "```"]
    lines += ["", MARKER]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trustbench.gate",
        description="Compare main's system with a pull request's, using committed results. "
        "Exit codes: 0 no significant regression, 1 significant regression, 2 could not run.",
    )
    parser.add_argument("dataset", type=Path, help="JSONL dataset file")
    parser.add_argument("--baseline", type=Path, required=True, help="main's system.toml")
    parser.add_argument("--candidate", type=Path, default=SYSTEM_FILE, help="the PR's system.toml")
    parser.add_argument("--markdown", type=Path, help="also write a report for a PR comment")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args(argv)
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")

    try:
        dataset = load_examples(args.dataset)
        paths = (read_system(args.baseline), read_system(args.candidate))
        baseline, candidate = read_results(paths[0]), read_results(paths[1])
        check_pair(baseline, candidate, dataset_version(dataset))
        comparison = None
        if baseline.answers != candidate.answers:
            examples = select_examples(dataset, baseline.split)
            comparison = compare_answers(
                examples, baseline.answers, candidate.answers, alpha=args.alpha
            )
    except (DatasetError, GateError, ResultsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}")
        return 2

    print(f"baseline   {paths[0]} ({baseline.model})")
    print(f"candidate  {paths[1]} ({candidate.model})")
    for name, old, new in changes(baseline, candidate):
        print(f"changed    {name}: {old} -> {new}")
    if comparison is None:
        print("verdict: NO SYSTEM CHANGE (the two systems give the same answers)")
    else:
        for line in report(comparison):
            print(line)
    if args.markdown is not None:
        args.markdown.write_text(markdown(paths, baseline, candidate, comparison), "utf-8")
    if comparison is not None and comparison.verdict is Verdict.REGRESSION:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
