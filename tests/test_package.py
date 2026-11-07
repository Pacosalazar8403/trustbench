"""Smoke tests: the package installs, imports, and reports one consistent version."""

from importlib.metadata import version

import pytest

import trustbench


def test_version_matches_installed_metadata() -> None:
    # pyproject.toml is the single source of truth for the version.
    assert trustbench.__version__ == version("trustbench")


def test_main_prints_name_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    trustbench.main()
    captured = capsys.readouterr()
    assert captured.out == f"trustbench {trustbench.__version__}\n"
