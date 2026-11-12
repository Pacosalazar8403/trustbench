"""Tests for the LLM judge, with a fake judge model."""

import hashlib
import json
from pathlib import Path

import pytest

from trustbench.corpus import build_corpus
from trustbench.judge import (
    JUDGE_TEMPLATES,
    build_judge_prompt,
    judge_answers,
    main,
    parse_verdict,
    summarize,
)
from trustbench.retrieval import build_index
from trustbench.runner import AnswerCache, Generation, OllamaModel, run
from trustbench.schema import Category, Difficulty, Evidence, Example

RIGHT = "Call 719-255-3460."
WRONG = "Call 719-255-0000."
DOCUMENTS = {"aid": "Call the financial aid office at 719-255-3460 for help with the FAFSA."}


def make_example(
    example_id: str, reference: str = RIGHT, category: Category = Category.LOOKUP
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


class FakeJudge:
    """Passes an answer when it contains the right phone number; counts its calls."""

    name = "fake-judge format=json"

    def __init__(self, reply: str | None = None) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> Generation:
        self.prompts.append(prompt)
        if self.reply is not None:
            return Generation(self.reply, tokens=1, seconds=0.5)
        graded = prompt.split("Answer to grade: ")[1].split("\n")[0]
        verdict = "correct" if "3460" in graded else "incorrect"
        return Generation(json.dumps({"reason": "checked", "verdict": verdict}), 9, 0.5)


def test_prompt_has_the_question_reference_evidence_and_answer() -> None:
    example = make_example("ex-001")
    prompt = build_judge_prompt(example, "Dial 719-255-3460.")
    assert f"Question: {example.question}\n" in prompt
    assert f"Reference answer: {RIGHT}\n" in prompt
    assert "- (aid) q\n" in prompt
    assert "Answer to grade: Dial 719-255-3460.\n" in prompt
    assert prompt.index('"reason"') < prompt.index('"verdict"')


def test_prompt_says_when_there_is_no_evidence() -> None:
    example = make_example("ex-001", "The documents don't say.", Category.UNANSWERABLE)
    assert "- (none: the documents do not answer this question)" in build_judge_prompt(example, "x")


@pytest.mark.parametrize(
    ("version", "digest"),
    [
        ("v1", "e913fa3b74b7d63540cb183447e8e2ce5f93f7a33b6170b3f4709c411113784f"),
        ("v2", "cd05c79eaca1f855fd2ac295e00c4538f186c9e2997748339f7391d77ec95954"),
    ],
)
def test_measured_prompts_are_frozen_so_their_judgments_stay_valid(
    version: str, digest: str
) -> None:
    assert hashlib.sha256(JUDGE_TEMPLATES[version].encode("utf-8")).hexdigest() == digest


def test_the_v2_prompt_states_its_rules_and_keeps_reason_before_verdict() -> None:
    example = make_example("ex-001")
    prompt = build_judge_prompt(example, "Dial 719-255-3460.", "v2")
    assert prompt != build_judge_prompt(example, "Dial 719-255-3460.", "v1")
    assert "even if the rest is right" in prompt
    assert "pointing the student to an office, is correct" in prompt
    assert "not a reason to mark the answer incorrect" in prompt
    assert 'the verdict must be "incorrect"' in prompt
    assert "Answer to grade: Dial 719-255-3460.\n" in prompt
    assert prompt.index('"reason"') < prompt.index('"verdict"')


def test_the_v3_prompt_requires_only_what_the_question_asks() -> None:
    prompt = build_judge_prompt(make_example("ex-001"), "Dial 719-255-3460.", "v3")
    assert "a detail the question does not ask about is not required" in prompt
    assert "not wrong just because the reference answer does not mention it" in prompt
    assert "Only in this case, making up steps" in prompt
    assert "invented" not in prompt.split("Answer to grade:")[1]
    assert prompt.index('"reason"') < prompt.index('"verdict"')


def test_an_unknown_prompt_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown judge prompt 'v9'"):
        build_judge_prompt(make_example("ex-001"), "x", "v9")


@pytest.mark.parametrize(
    ("reply", "passed"),
    [
        ('{"reason": "same meaning", "verdict": "correct"}', True),
        ('{"reason": "wrong office", "verdict": "incorrect"}', False),
        ('{"verdict": " Correct "}', True),
        ('{"reason": "unsure", "verdict": "partly"}', None),
        ('{"reason": "no verdict at all"}', None),
        ('["correct"]', None),
        ("The answer is correct.", None),
    ],
)
def test_parse_verdict(reply: str, passed: bool | None) -> None:
    assert parse_verdict(reply).passed is passed


def test_parse_verdict_keeps_the_reason() -> None:
    assert parse_verdict('{"reason": "wrong office", "verdict": "incorrect"}').reason == (
        "wrong office"
    )


def test_judgments_are_cached_and_paired_with_the_scorers(tmp_path: Path) -> None:
    examples = [make_example("ex-001"), make_example("ex-002")]
    answers = {"ex-001": RIGHT, "ex-002": WRONG}
    judge = FakeJudge()

    first = judge_answers(examples, answers, judge, AnswerCache(tmp_path / "j.jsonl"), "v1")
    second = judge_answers(examples, answers, judge, AnswerCache(tmp_path / "j.jsonl"), "v1")

    assert [judgment.judge.passed for judgment in first] == [True, False]
    assert [judgment.scorer.passed for judgment in first] == [True, False]
    assert [judgment.cached for judgment in second] == [True, True]
    assert len(judge.prompts) == 2


def test_a_different_answer_is_judged_again(tmp_path: Path) -> None:
    examples = [make_example("ex-001")]
    cache = AnswerCache(tmp_path / "j.jsonl")
    judge = FakeJudge()
    judge_answers(examples, {"ex-001": RIGHT}, judge, cache, "v1")
    assert not judge_answers(examples, {"ex-001": WRONG}, judge, cache, "v1")[0].cached


def test_summary_reports_agreement_prose_and_disagreements(tmp_path: Path) -> None:
    examples = [make_example(f"ex-00{n}") for n in (1, 2, 3)]
    prose = make_example("ex-009", "Contact your advisor.")
    answers = {"ex-001": RIGHT, "ex-002": WRONG, "ex-003": RIGHT, "ex-009": "Ask 3460."}
    judgments = judge_answers(
        [*examples, prose], answers, FakeJudge(), AnswerCache(tmp_path / "j.jsonl"), "v1"
    )
    lines = summarize(judgments)

    assert lines[0] == "4 judgments, 0 without a valid verdict"
    assert "on 3 examples both can judge: agree on 3 of 3 (100%), kappa 1.00" in lines[1]
    assert "judge passed 1 of 1 answers the scorers cannot judge (100%)" in lines


def test_summary_lists_answers_without_a_verdict(tmp_path: Path) -> None:
    judgments = judge_answers(
        [make_example("ex-001")],
        {"ex-001": RIGHT},
        FakeJudge(reply="not json"),
        AnswerCache(tmp_path / "j.jsonl"),
        "v1",
    )
    lines = summarize(judgments)
    assert lines[0] == "1 judgments, 1 without a valid verdict"
    assert lines[-1].startswith("  no verdict ex-001: judge reply is not JSON")


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


class FakeSystem:
    def __init__(self, reply: str) -> None:
        self.name = OllamaModel("sys").name
        self.reply = reply

    def generate(self, prompt: str) -> Generation:
        return Generation(self.reply, tokens=1, seconds=0.1)


def test_main_judges_cached_answers_and_reuses_judgments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples = [make_example("ex-001"), make_example("ex-002", "Contact your advisor.")]
    dataset, corpus = write_project(tmp_path, examples)
    run(examples, build_index(DOCUMENTS), FakeSystem(RIGHT), AnswerCache(tmp_path / "a.jsonl"), "v")
    judge = FakeJudge()
    argv = [str(dataset), str(corpus), "--model", "sys", "--split", "all"]
    argv += ["--cache", str(tmp_path / "a.jsonl"), "--judgments", str(tmp_path / "j.jsonl")]

    assert main(argv, make_judge=lambda model, url: judge) == 0
    output = capsys.readouterr().out
    assert "judge   fake-judge format=json" in output
    assert "[1/2] ex-001 pass (new)" in output
    assert "judge passed 1 of 1 answers the scorers cannot judge (100%)" in output

    assert main(argv, make_judge=lambda model, url: judge) == 0
    assert "[2/2] ex-002 pass (cached)" in capsys.readouterr().out
    assert len(judge.prompts) == 2


class InterruptedJudge(FakeJudge):
    """Judges one answer, then behaves as if the user pressed Ctrl+C."""

    def generate(self, prompt: str) -> Generation:
        if self.prompts:
            raise KeyboardInterrupt
        return super().generate(prompt)


def test_ctrl_c_stops_cleanly_and_keeps_finished_judgments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples = [make_example("ex-001"), make_example("ex-002")]
    dataset, corpus = write_project(tmp_path, examples)
    run(examples, build_index(DOCUMENTS), FakeSystem(RIGHT), AnswerCache(tmp_path / "a.jsonl"), "v")
    argv = [str(dataset), str(corpus), "--model", "sys", "--split", "all"]
    argv += ["--cache", str(tmp_path / "a.jsonl"), "--judgments", str(tmp_path / "j.jsonl")]

    assert main(argv, make_judge=lambda model, url: InterruptedJudge()) == 130
    assert f"stopped; 1 judgments are saved in {tmp_path / 'j.jsonl'}" in capsys.readouterr().out
    assert len(AnswerCache(tmp_path / "j.jsonl")) == 1


def test_main_uses_the_chosen_prompt_and_caches_each_version_separately(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples = [make_example("ex-001")]
    dataset, corpus = write_project(tmp_path, examples)
    run(examples, build_index(DOCUMENTS), FakeSystem(RIGHT), AnswerCache(tmp_path / "a.jsonl"), "v")
    judge = FakeJudge()
    argv = [str(dataset), str(corpus), "--model", "sys", "--split", "all"]
    argv += ["--cache", str(tmp_path / "a.jsonl"), "--judgments", str(tmp_path / "j.jsonl")]

    assert main(argv, make_judge=lambda model, url: judge) == 0
    assert main([*argv, "--prompt", "v2"], make_judge=lambda model, url: judge) == 0

    output = capsys.readouterr().out
    assert "judge   fake-judge format=json, prompt v1" in output
    assert "judge   fake-judge format=json, prompt v2" in output
    assert output.count("ex-001 pass (new)") == 2
    assert "Grade by these rules" in judge.prompts[1]


def test_main_returns_2_when_answers_are_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, corpus = write_project(tmp_path, [make_example("ex-001")])
    argv = [str(dataset), str(corpus), "--model", "sys", "--split", "all"]
    argv += ["--cache", str(tmp_path / "a.jsonl"), "--judgments", str(tmp_path / "j.jsonl")]

    assert main(argv, make_judge=lambda model, url: FakeJudge()) == 2
    assert "no cached answer from sys" in capsys.readouterr().out
