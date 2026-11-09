"""Consistency checks for the committed dataset and corpus files.

The full evidence check needs the corpus text, which is not committed, so it runs
locally. These tests check everything that can be checked from committed files alone,
so they run in CI on every pull request.
"""

from pathlib import Path

from trustbench.corpus import read_manifest, read_sources
from trustbench.dataset import load_examples
from trustbench.splits import find_evidence_conflicts

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "uccs.jsonl"
CORPUS = ROOT / "corpus" / "uccs"


def test_dataset_loads() -> None:
    assert load_examples(DATASET)


def test_every_evidence_doc_id_is_a_corpus_source() -> None:
    sources = {source.doc_id for source in read_sources(CORPUS / "sources.txt")}
    cited = {evidence.doc_id for ex in load_examples(DATASET) for evidence in ex.evidence}

    assert cited - sources == set()


def test_sources_and_manifest_list_the_same_documents() -> None:
    sources = [source.doc_id for source in read_sources(CORPUS / "sources.txt")]
    manifest = [entry.doc_id for entry in read_manifest(CORPUS / "manifest.jsonl")]

    assert sources == manifest


def test_examples_that_share_evidence_share_a_group() -> None:
    conflicts = find_evidence_conflicts(load_examples(DATASET))

    assert conflicts == [], "\n".join(str(conflict) for conflict in conflicts)
