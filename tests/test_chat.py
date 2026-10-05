"""Tests for Ghana Chat retrieval, abbreviation resolution, multi-turn follow-up detection, and API endpoints."""

import pytest
from ghana_chat.retriever import HeadKGRetriever
from ghana_chat.abbreviations import AbbreviationResolver
from ghana_chat import config


def test_abbreviation_resolver():
    resolver = AbbreviationResolver()
    pairs, aliases = resolver.resolve_query("Who is the chairperson of the NDPC?")
    assert len(pairs) > 0
    assert ("NDPC", "National Development Planning Commission") in pairs
    assert "ndpc" in aliases
    assert "national development planning commission" in aliases


def test_abbreviation_bidirectional():
    resolver = AbbreviationResolver()
    pairs, aliases = resolver.resolve_query("What does the Electoral Commission do?")
    assert any(p[0] == "EC" for p in pairs)
    assert "ec" in aliases
    assert "electoral commission" in aliases


def test_retriever_initialization():
    retriever = HeadKGRetriever(config.KG_PATH)
    assert len(retriever.heads_map) > 0


def test_head_focused_retrieval_with_abbreviation():
    retriever = HeadKGRetriever(config.KG_PATH)
    
    # Query with acronym NDPC
    res = retriever.retrieve("Who is the chairperson of the NDPC?")
    triples = res.get("triples", [])
    assert len(triples) > 0
    # Should find facts under NDPC
    heads = [t["head"].lower() for t in triples]
    assert any("ndpc" in h for h in heads)


def test_multiturn_followup_detection_no_new_entities():
    retriever = HeadKGRetriever(config.KG_PATH)
    
    # Turn 1: User asks about NDPC
    t1 = retriever.process_turn("Who is the chairperson of the NDPC?")
    assert t1["is_followup"] is False
    assert len(t1["new_triples"]) > 0
    assert len(t1["all_entities"]) > 0
    assert len(t1.get("abbreviations", [])) > 0

    # Turn 2: User asks follow-up with pronoun ("Where is it located?")
    t2 = retriever.process_turn(
        query="Where is it located?",
        history_entities=t1["all_entities"],
        existing_triples=t1["all_triples"]
    )
    assert t2["is_followup"] is True
    assert len(t2["new_triples"]) == 0
    assert len(t2["all_triples"]) == len(t1["all_triples"])


def test_no_reflexive_triples():
    retriever = HeadKGRetriever(config.KG_PATH)
    for head_key, items in retriever.heads_map.items():
        for t in items:
            assert t["head"].lower() != t["tail"].lower(), f"Reflexive triple found: {t}"
