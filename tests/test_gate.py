"""Tests for the pull request gate, on small datasets and results files."""

from pathlib import Path

import pytest

from trustbench.dataset import dataset_version, load_examples
from trustbench.gate import MARKER, SYSTEM_FILE, GateError, main, read_system
from trustbench.results import Results, read_results, write_results
from trustbench.schema import Category, Difficulty, Evidence, Example

RIGHT = "Call 719-255-3460."
WRONG = "Call 719-255-0000."


def make_example(example_id: str) -> Example:
    return Example(
        id=example_id,
        question=f"What is the financial aid phone number? ({example_id})",
        reference_answer=RIGHT,
        category=Category.LOOKUP,
        difficulty=Difficulty.EASY,
        evidence=(Evidence(doc_id="aid", quote="q"),),
    )


class Project:
    """A dataset, results files, and system files in a temporary folder."""

    def __init__(self, tmp_path: Path, count: int = 10) -> None:
        self.root = tmp_path
        self.examples = [make_example(f"ex-{number:03d}") for number in range(1, count + 1)]
        self.dataset = tmp_path / "data.jsonl"
        self.dataset.write_text(
            "".join(example.model_dump_json() + "\n" for example in self.examples), "utf-8"
        )
        self.version = dataset_version(self.examples)

    def results(self, name: str, wrong: int, **changes: object) -> Path:
        """Write results that get the first `wrong` examples wrong; return a system file."""
        answers = {
            example.id: WRONG if position < wrong else RIGHT
            for position, example in enumerate(self.examples)
        }
        fields: dict[str, object] = {
            "model": f"ollama:{name} temperature=0.0 seed=0",
            "split": "all",
            "k": 5,
            "dataset_version": self.version,
            "prompt_sha256": "p" * 64,
            "corpus_sha256": "c" * 64,
            "answers": answers,
        }
        path = self.root / "results" / f"all-{name}.json"
        write_results(Results.model_validate(fields | changes), path)
        system = self.root / f"{name}.toml"
        system.write_text(f'[system]\nresults = "{path}"\n', encoding="utf-8")
        return system

    def gate(self, baseline: Path, candidate: Path, *extra: str) -> int:
        argv = [str(self.dataset), "--baseline", str(baseline), "--candidate", str(candidate)]
        return main([*argv, *extra])


def test_a_system_file_names_its_results(tmp_path: Path) -> None:
    path = tmp_path / "system.toml"
    path.write_text('[system]\nresults = "results/dev-m.json"\n', encoding="utf-8")
    assert read_system(path) == Path("results/dev-m.json")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[system]\n", "needs a \\[system\\] table"),
        ('results = "x.json"\n', "needs a \\[system\\] table"),
        ("[system\n", "not valid TOML"),
    ],
)
def test_a_bad_system_file_is_refused(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "system.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(GateError, match=message):
        read_system(path)


def test_the_same_system_is_not_tested(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = Project(tmp_path)
    system = project.results("base", wrong=3)
    report = tmp_path / "report.md"

    assert project.gate(system, system, "--markdown", str(report)) == 0
    assert "verdict: NO SYSTEM CHANGE" in capsys.readouterr().out
    text = report.read_text(encoding="utf-8")
    assert text.startswith("## TrustBench gate: ⚪ NO SYSTEM CHANGE")
    assert text.rstrip().endswith(MARKER)


def test_a_significant_regression_fails_and_names_what_changed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = Project(tmp_path)
    baseline = project.results("good", wrong=0)
    candidate = project.results("bad", wrong=7, k=3)
    report = tmp_path / "report.md"

    assert project.gate(baseline, candidate, "--markdown", str(report)) == 1
    output = capsys.readouterr().out
    assert "changed    model: ollama:good temperature=0.0 seed=0 -> ollama:bad" in output
    assert "changed    k: 5 -> 3" in output
    assert "verdict: REGRESSION" in output
    text = report.read_text(encoding="utf-8")
    assert text.startswith("## TrustBench gate: ❌ REGRESSION")
    assert "| k | `5` | `3` |" in text
    assert "lost (7): ex-001, ex-002" in text
    assert "| prompt |" not in text


def test_an_improvement_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = Project(tmp_path)
    assert project.gate(project.results("bad", wrong=7), project.results("good", wrong=0)) == 0
    assert "verdict: IMPROVEMENT" in capsys.readouterr().out


def test_stale_results_stop_the_gate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = Project(tmp_path)
    baseline = project.results("base", wrong=0)
    candidate = project.results("old", wrong=0, dataset_version="0123456789abcdef")

    assert project.gate(baseline, candidate) == 2
    assert "candidate results are stale: made for dataset 0123456789abcdef" in (
        capsys.readouterr().out
    )


def test_different_splits_stop_the_gate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = Project(tmp_path)
    baseline = project.results("base", wrong=0)
    candidate = project.results("test", wrong=0, split="test")

    assert project.gate(baseline, candidate) == 2
    assert "baseline is on all but candidate is on test" in capsys.readouterr().out


def test_a_missing_answer_stops_the_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = Project(tmp_path)
    baseline = project.results("base", wrong=0)
    candidate = project.results("short", wrong=1, answers={"ex-001": RIGHT})

    assert project.gate(baseline, candidate) == 2
    assert "no candidate answer for ex-002" in capsys.readouterr().out


def test_a_missing_results_file_stops_the_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = Project(tmp_path)
    missing = tmp_path / "missing.toml"
    missing.write_text('[system]\nresults = "results/none.json"\n', encoding="utf-8")

    assert project.gate(project.results("base", wrong=0), missing) == 2
    assert "no such results file" in capsys.readouterr().out


def test_the_committed_system_names_current_results() -> None:
    if not SYSTEM_FILE.exists():
        pytest.skip("no system.toml committed yet")
    results = read_results(read_system(SYSTEM_FILE))
    assert results.dataset_version == dataset_version(load_examples(Path("data/uccs.jsonl")))
