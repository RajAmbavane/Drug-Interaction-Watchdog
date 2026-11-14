"""Tests for the unified text chunker used to build the RAG corpus."""

import pytest

from ingestion.text_chunker import (
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    TextChunker,
)


@pytest.fixture
def chunker():
    return TextChunker()


def test_clean_strips_control_characters_and_whitespace():
    assert TextChunker._clean("  Warfarin\x00 may\n\n increase\tbleeding  ") == "Warfarin may increase bleeding"


def test_sentence_split_keeps_dose_abbreviations_intact():
    text = "Give 5 mg. daily with food. Dr. Smith reviewed it. Monitor INR weekly."
    sentences = TextChunker._sentence_split(text)
    assert sentences == [
        "Give 5 mg. daily with food.",
        "Dr. Smith reviewed it.",
        "Monitor INR weekly.",
    ]


def test_short_text_below_minimum_is_dropped(chunker):
    assert chunker._split("Too short.") == []


def test_medium_text_is_a_single_chunk(chunker):
    text = "Concomitant use increases the risk of serious bleeding events. " * 3
    chunks = chunker._split(text)
    assert len(chunks) == 1
    assert MIN_CHUNK_CHARS <= len(chunks[0]) <= MAX_CHUNK_CHARS


def test_long_text_is_split_with_sentence_overlap(chunker):
    sentences = [f"Sentence number {i} describes an interaction finding in detail." for i in range(60)]
    chunks = chunker._split(" ".join(sentences))

    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)
    # The last sentence of one chunk opens the next one
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = prev.rsplit(". ", 1)[-1]
        assert nxt.startswith(last_sentence.rstrip("."))
    # Nothing is lost
    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)


def test_make_chunk_fills_metadata(chunker):
    chunk = chunker._make_chunk(
        text="  Warfarin  and aspirin increase bleeding risk.  ",
        source="dailymed",
        doc_id="set-1",
        drug_names=["warfarin", "", "  ", "aspirin"],
        section="drug_interactions",
        priority=True,
    )
    assert chunk.text == "Warfarin and aspirin increase bleeding risk."
    assert chunk.drug_names == ["warfarin", "aspirin"]
    assert chunk.char_count == len(chunk.text)
    assert chunk.token_estimate >= 1


def test_identical_text_gets_identical_hash(chunker):
    a = chunker._make_chunk("Same TEXT here.", "faers", "1", [], "x", False)
    b = chunker._make_chunk("same text   here.", "pubmed", "2", [], "y", False)
    assert a.content_hash == b.content_hash
    assert a.chunk_id != b.chunk_id


def test_extract_drug_names_from_pubmed_query():
    q = '"warfarin"[Title] AND "aspirin"[Title]'
    assert TextChunker._extract_drug_names_from_query(q) == ["warfarin", "aspirin"]


def test_results_require_run(chunker):
    with pytest.raises(RuntimeError):
        chunker.get_chunks_df()
