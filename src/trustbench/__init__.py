"""TrustBench: a statistical regression gate for LLM and RAG systems."""

from importlib.metadata import version

__version__ = version("trustbench")

__all__ = ["__version__", "main"]


def main() -> None:
    """Entry point for the `trustbench` command (placeholder until the real CLI lands)."""
    print(f"trustbench {__version__}")