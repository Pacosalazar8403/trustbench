"""Tests for comparing two systems' answers and the regression verdict."""

from pathlib import Path

import pytest

from trustbench.compare import Verdict, compare_answers, load_answers, main, report
from trustbench.corpus import build_corpus
from trustbench.retrieval import build_index
from trustbench.runner import AnswerCache, Generation, OllamaModel, run
from trustbench.schema import Category, Difficulty, Evidence, Example

RIGHT = "Call 719-255-3460."
WRONG = "Call 719-255-0000."


def make_example(
    example_id: str,
    reference: str = RIGHT,
    category: Category = Category.LOOKUP,
) -> Example:
    evidence = () if category is Category.UNANSWERABLE else (Evidence(doc_id="aid", quote="q"),)
    return Example(
        id=example_id,
        question=f"What is the financial aid phone number? ({example_id})",
        reference_answer=reference,
        category=category,
        difficulty=Difficulty.EASY,
        evidence=evidence,
    )


def examples_and_answers(
    both: int, only_baseline: int, only_candidate: int, neither: int
) -> tuple[list[Example], dict[str, str], dict[str, str]]:
    """Judged examples where the two systems pass in the given combination."""
    examples: list[Example] = []
    baseline: dict[str, str] = {}
    candidate: dict[str, str] = {}
    plan = [(True, True)] * both + [(True, False)] * only_baseline
    plan += [(False, True)] * only_candidate + [(False, False)] * neither
    for number, (old, new) in enumerate(plan, start=1):
        example = make_example(f"ex-{number:03d}")
        examples.append(example)
        baseline[example.id] = RIGHT if old else WRONG
        candidate[example.id] = RIGHT if new else WRONG
    return examples, baseline, candidate


def test_pairs_count_wins_losses_and_pass_rates() -> None:
    examples, baseline, candidate = examples_and_answers(5, 1, 3, 2)
    comparison = compare_answers(examples, baseline, candidate)

    assert len(comparison.pairs) == 11
    assert (comparison.overall.wins, comparison.overall.losses) == (3, 1)
    assert sum(pair.baseline for pair in comparison.pairs) == 6
    assert sum(pair.candidate for pair in comparison.pairs) == 8
    assert comparison.disagreement_rate == pytest.approx(4 / 11)


def test_a_one_sided_set_of_losses_is_a_regression() -> None:
    examples, baseline, candidate = examples_and_answers(20, 8, 0, 5)
    comparison = compare_answers(examples, baseline, candidate)
    assert comparison.overall.p_value <= 0.05
    assert comparison.verdict is Verdict.REGRESSION
    assert comparison.interval.high < 0


def test_a_one_sided_set_of_wins_is_an_improvement() -> None:
    examples, baseline, candidate = examples_and_answers(20, 0, 8, 5)
    assert compare_answers(examples, baseline, candidate).verdict is Verdict.IMPROVEMENT


def test_five_losses_and_no_wins_are_not_enough_to_fail_the_gate() -> None:
    examples, baseline, candidate = examples_and_answers(20, 5, 0, 5)
    comparison = compare_answers(examples, baseline, candidate)
    assert comparison.overall.p_value == pytest.approx(0.0625)
    assert comparison.verdict is Verdict.NO_SIGNIFICANT_CHANGE


def test_identical_systems_show_no_change() -> None:
    examples, baseline, _ = examples_and_answers(6, 0, 0, 4)
    comparison = compare_answers(examples, baseline, baseline)
    assert comparison.verdict is Verdict.NO_SIGNIFICANT_CHANGE
    assert comparison.disagreement_rate == 0


def test_alpha_decides_the_verdict() -> None:
    examples, baseline, candidate = examples_and_answers(20, 7, 0, 5)
    assert compare_answers(examples, baseline, candidate).verdict is Verdict.REGRESSION
    strict = compare_answers(examples, baseline, candidate, alpha=0.01)
    assert strict.verdict is Verdict.NO_SIGNIFICANT_CHANGE


def test_examples_the_scorers_cannot_judge_are_counted_but_not_compared() -> None:
    examples, baseline, candidate = examples_and_answers(3, 0, 1, 0)
    prose = make_example("ex-900", reference="Contact your academic advisor.")
    comparison = compare_answers(
        [*examples, prose], baseline | {prose.id: "a"}, candidate | {prose.id: "b"}
    )
    assert len(comparison.pairs) == 4
    assert comparison.not_judged == 1


def test_slices_are_per_category_and_holm_adjusted() -> None:
    examples, baseline, candidate = examples_and_answers(4, 0, 6, 0)
    unanswerable = [
        make_example(f"un-{number}", "The documents don't say.", Category.UNANSWERABLE)
        for number in (1, 2)
    ]
    for example in unanswerable:
        baseline[example.id] = "It is $50."
        candidate[example.id] = "The documents don't say."
    comparison = compare_answers([*examples, *unanswerable], baseline, candidate)

    assert [part.name for part in comparison.slices] == ["lookup", "unanswerable"]
    lookup, other = comparison.slices
    assert (lookup.test.wins, other.test.wins) == (6, 2)
    assert lookup.adjusted_p == pytest.approx(2 * lookup.test.p_value)
    assert other.adjusted_p >= other.test.p_value


def test_a_missing_answer_or_nothing_to_judge_is_an_error() -> None:
    examples, baseline, candidate = examples_and_answers(2, 0, 0, 0)
    del candidate["ex-002"]
    with pytest.raises(ValueError, match="no candidate answer for ex-002"):
        compare_answers(examples, baseline, candidate)
    prose = make_example("ex-900", reference="Contact your advisor.")
    with pytest.raises(ValueError, match="none of the examples can be judged"):
        compare_answers([prose], {prose.id: "a"}, {prose.id: "b"})


def test_report_ends_with_the_verdict_and_names_what_changed() -> None:
    examples, baseline, candidate = examples_and_answers(20, 8, 1, 5)
    lines = report(compare_answers(examples, baseline, candidate))

    assert lines[0] == "34 judged examples (0 more need a judge)"
    assert "lost (8): ex-021, ex-022" in "\n".join(lines)
    assert "won (1): ex-029" in lines
    assert any(line.startswith("power: at 34 examples and a 26.5% disagreement") for line in lines)
    assert any("cover less often than they claim" in line for line in lines)
    assert lines[-1].startswith("verdict: REGRESSION (p = 0.0391 <= alpha 0.05)")


class FakeModel:
    """Answers every prompt with one text, under a real Ollama model's name."""

    def __init__(self, model: str, reply: str) -> None:
        self.name = OllamaModel(model).name
        self.reply = reply

    def generate(self, prompt: str) -> Generation:
        return Generation(self.reply, tokens=1, seconds=0.1)


DOCUMENTS = {"aid": "Call the financial aid office at 719-255-3460 for help with the FAFSA."}


def write_project(tmp_path: Path, examples: list[Example]) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    (corpus / "raw").mkdir(parents=True)
    (corpus / "sources.txt").write_text("aid https://example.edu/aid\n", encoding="utf-8")
    (corpus / "raw" / "aid.html").write_text(
        f"<main><p>{DOCUMENTS['aid']}</p></main>", encoding="utf-8"
    )
    build_corpus(corpus)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "".join(example.model_dump_json() + "\n" for example in examples), encoding="utf-8"
    )
    return dataset, corpus


def test_load_answers_finds_cached_answers_and_reports_misses(tmp_path: Path) -> None:
    examples = [make_example("ex-001"), make_example("ex-002")]
    index = build_index(DOCUMENTS)
    cache = AnswerCache(tmp_path / "answers.jsonl")
    run(examples[:1], index, FakeModel("base", RIGHT), cache, "v1")

    answers, missing = load_answers(examples, index, cache, OllamaModel("base").name, 5)
    assert answers == {"ex-001": RIGHT}
    assert missing == ["ex-002"]


@pytest.mark.parametrize(
    ("candidate_reply", "exit_code", "verdict"),
    [(WRONG, 1, "REGRESSION"), (RIGHT, 0, "NO SIGNIFICANT CHANGE")],
)
def test_main_fails_only_on_a_significant_regression(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    candidate_reply: str,
    exit_code: int,
    verdict: str,
) -> None:
    examples = [make_example(f"ex-{number:03d}") for number in range(1, 9)]
    dataset, corpus = write_project(tmp_path, examples)
    cache = AnswerCache(tmp_path / "answers.jsonl")
    index = build_index(DOCUMENTS)
    run(examples, index, FakeModel("base", RIGHT), cache, "v1")
    run(examples, index, FakeModel("cand", candidate_reply), cache, "v1")
    argv = [str(dataset), str(corpus), "--baseline", "base", "--candidate", "cand"]
    argv += ["--split", "all", "--cache", str(tmp_path / "answers.jsonl")]

    assert main(argv) == exit_code
    output = capsys.readouterr().out
    assert "baseline   ollama:base temperature=0.0 seed=0" in output
    assert f"verdict: {verdict}" in output


def test_main_returns_2_and_says_how_to_fix_missing_answers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples = [make_example("ex-001"), make_example("ex-002")]
    dataset, corpus = write_project(tmp_path, examples)
    cache = AnswerCache(tmp_path / "answers.jsonl")
    run(examples, build_index(DOCUMENTS), FakeModel("base", RIGHT), cache, "v1")
    argv = [str(dataset), str(corpus), "--baseline", "base", "--candidate", "cand"]
    argv += ["--split", "all", "--cache", str(tmp_path / "answers.jsonl")]

    assert main(argv) == 2
    output = capsys.readouterr().out
    assert "error: 2 of 2 all examples have no cached answer from cand (first: ex-001)" in output
    assert "--model cand --split all --k 5" in output
