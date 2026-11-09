"""Pydantic models for one TrustBench dataset example.

An example is a question, the answer a careful human would give, and the evidence that
proves it. Everything is validated here, at the boundary, so the rest of TrustBench can
trust the data instead of re-checking it.
"""

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

# Text that must contain something besides whitespace. Surrounding whitespace is removed.
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# Lowercase words separated by single hyphens, like "withdraw-001".
ExampleId = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=64),
]


class Category(StrEnum):
    """The kind of question, which decides what the example tests."""

    LOOKUP = "lookup"
    PARAPHRASE = "paraphrase"
    MULTI_PASSAGE = "multi_passage"
    CONDITIONAL = "conditional"
    FALSE_PREMISE = "false_premise"
    UNANSWERABLE = "unanswerable"


class Difficulty(StrEnum):
    """How hard the question is for a careful human reading the documents."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Evidence(BaseModel):
    """An exact quote from a source document that supports the reference answer.

    Evidence points at text, not at a chunk ID, so it stays valid no matter how a RAG
    system splits documents into chunks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: NonEmptyText
    quote: NonEmptyText


class Example(BaseModel):
    """One question with its reference answer and supporting evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: ExampleId
    question: NonEmptyText
    reference_answer: NonEmptyText
    category: Category
    difficulty: Difficulty
    evidence: tuple[Evidence, ...] = ()
    # Examples with the same group always land in the same split. Give related examples
    # (for example, two questions that quote the same sentence) one shared group, so a
    # near-copy of a test question can never end up in dev. No group means "my own group".
    group: ExampleId | None = None

    @property
    def answerable(self) -> bool:
        """Whether the documents contain the answer. Derived from the category."""
        return self.category is not Category.UNANSWERABLE

    @model_validator(mode="after")
    def _check_evidence_fits_category(self) -> Self:
        if self.answerable and not self.evidence:
            raise ValueError("answerable examples need at least one piece of evidence")
        if not self.answerable and self.evidence:
            raise ValueError("unanswerable examples must not have evidence")
        if self.category is Category.MULTI_PASSAGE and len(self.evidence) < 2:
            raise ValueError("multi_passage examples need at least two pieces of evidence")
        return self
