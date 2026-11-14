"""Tests for retrieval fusion and citation tracking in the RAG layer."""

import pytest

from tests.conftest import load_module
from rag.citation_tracker import (
    CitationTracker,
    clean_snippet,
    infer_doc_url,
    normalize_source,
    stable_source_id,
)

retriever = load_module(
    "rag/retriever.py",
    stubs=("faiss", "psycopg2", "rank_bm25", "transformers"),
)


# ── Reciprocal rank fusion ───────────────────────────────────────────────────

def test_rrf_rewards_documents_found_by_both_retrievers():
    fused = retriever.reciprocal_rank_fusion(["a", "b", "c"], ["c", "a", "d"], k=60)
    ids = [cid for cid, _ in fused]
    # "a" ranks 1st and 2nd, "c" ranks 3rd and 1st — both beat single-list hits
    assert set(ids[:2]) == {"a", "c"}
    assert ids.index("a") < ids.index("b")
    assert ids.index("c") < ids.index("d")


def test_rrf_score_matches_formula():
    fused = dict(retriever.reciprocal_rank_fusion(["x"], ["x"], k=60))
    assert fused["x"] == pytest.approx(2 / 61)


def test_rrf_handles_empty_lists():
    assert retriever.reciprocal_rank_fusion([], []) == []


def test_rrf_output_is_sorted_descending():
    fused = retriever.reciprocal_rank_fusion(list("abcdef"), list("fedcba"))
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_bm25_tokeniser_lowercases_and_splits():
    assert retriever.BM25Index._tokenise("Warfarin  and Aspirin") == ["warfarin", "and", "aspirin"]


# ── Citation helpers ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw, expected",
    [("DailyMed", "dailymed"), ("fda_label", "dailymed"), ("pmc", "pubmed"), ("", "unknown"), (None, "unknown")],
)
def test_normalize_source(raw, expected):
    assert normalize_source(raw) == expected


def test_clean_snippet_collapses_whitespace_and_truncates():
    assert clean_snippet("  a \n\n b\t c ") == "a b c"
    long = clean_snippet("x" * 500, max_chars=50)
    assert len(long) <= 53 and long.endswith("...")


def test_infer_doc_url_for_known_sources():
    assert infer_doc_url("pubmed", "PMID: 12345678") == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    assert infer_doc_url("drugbank", "DB00682") == "https://go.drugbank.com/drugs/DB00682"
    assert infer_doc_url("drugbank", "warfarin") == ""
    assert "setid=abc" in infer_doc_url("dailymed", "abc")
    assert infer_doc_url("pubmed", "1", existing_url="https://example.org") == "https://example.org"


def test_stable_source_id_groups_chunks_from_same_section():
    a = stable_source_id("dailymed", "SET1", "drug interactions", chunk_id="c1")
    b = stable_source_id("dailymed", "set1", "DRUG INTERACTIONS", chunk_id="c2")
    assert a == b


# ── CitationTracker ──────────────────────────────────────────────────────────

def test_tracker_assigns_keys_per_source_and_dedupes():
    t = CitationTracker()
    k1 = t.add(source="dailymed", doc_id="set1", section="WARNINGS", snippet="bleeding risk")
    k2 = t.add(source="dailymed", doc_id="set1", section="WARNINGS", snippet="duplicate", relevance_score=0.9)
    k3 = t.add(source="pubmed", doc_id="12345678", snippet="case report")
    k4 = t.add(source="dailymed", doc_id="set2", section="WARNINGS")

    assert (k1, k2, k3, k4) == ("[FDA-1]", "[FDA-1]", "[PUB-1]", "[FDA-2]")
    assert len(t) == 3
    # Duplicate keeps the first snippet but takes the higher relevance score
    first = t.to_key_map()["[FDA-1]"]
    assert first["snippet"] == "bleeding risk"
    assert first["relevance_score"] == 0.9


def test_tracker_validates_generated_text():
    t = CitationTracker()
    t.add(source="dailymed", doc_id="set1")
    t.add(source="faers", doc_id="q1")
    result = t.validate_text("Bleeding risk rises [FDA-1]. See also [PUB-7].")
    assert result.used_keys == ["[FDA-1]", "[PUB-7]"]
    assert result.missing_keys == ["[PUB-7]"]
    assert result.unused_keys == ["[FAERS-1]"]
    assert not result.ok


def test_tracker_format_references_when_empty():
    assert "no citations" in CitationTracker().format_references()
