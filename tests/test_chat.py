"""Tests for Ghana Chat retrieval, abbreviation resolution, pronoun follow-ups, and context isolation."""

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
    heads = [t["head"].lower() for t in triples]
    assert any("ndpc" in h for h in heads)


def test_multiturn_followup_detection_pronoun_and_attributes():
    """Verify that 'how many children does he have and does he have a wife?' is recognized

    as a follow-up on the established subject (he = Asiedu Nketia), NOT as new standalone
    topics for 'children' or 'wife'.
    """
    retriever = HeadKGRetriever(config.KG_PATH)

    # Turn 1
    t1 = retriever.process_turn("Who is Asiedu Nketia?")
    assert t1["is_followup"] is False

    # Turn 2: Follow-up using pronoun 'he' and attribute queries 'children', 'wife'
    t2 = retriever.process_turn(
        query="how many children does he have and does he have a wife?",
        history_entities=t1["all_entities"],
        existing_triples=t1["all_triples"]
    )

    # Must be recognized as follow-up
    assert t2["is_followup"] is True
    # Must NOT treat children/wife as new independent topic entities
    assert len(t2["new_entities"]) == 0
    # Must preserve the subject's context
    assert len(t2["active_triples"]) > 0
    for t in t2["active_triples"]:
        assert t["head"].lower() not in ("wife", "child", "children")


def test_topic_switch_with_new_entities():
    """Verify that asking a completely new question (e.g.

    cassava farming) after a political query triggers a topic switch and retrieves
    farming facts.
    """
    retriever = HeadKGRetriever(config.KG_PATH)

    # Turn 1: Politics
    t1 = retriever.process_turn("Who is Asiedu Nketia?")

    # Turn 2: Agriculture (Topic Switch)
    t2 = retriever.process_turn(
        query="in what month do ghanain farmers plant cassava",
        history_entities=t1["all_entities"],
        existing_triples=t1["all_triples"]
    )

    # Must recognize topic switch
    assert t2["is_followup"] is False
    # Must identify new agricultural entities
    assert any("cassava" in e.lower() or "farmer" in e.lower() for e in t2["new_entities"])
    # Active triples must relate to agriculture, not Asiedu Nketia
    for t in t2["active_triples"]:
        assert "asiedu nketia" not in t["head"].lower()


def test_no_reflexive_triples():
    retriever = HeadKGRetriever(config.KG_PATH)
    for head_key, items in retriever.heads_map.items():
        for t in items:
            assert t["head"].lower() != t["tail"].lower(), f"Reflexive triple found: {t}"


def test_sentence_server_anaphora():
    from ghana_chat.server import ChatMessage, is_followup_query, retrieval_query

    h = [ChatMessage(role="user", content="Who was Ghana's first president?")]
    
    # Follow-up with pronoun
    assert is_followup_query("When did he establish the CPP?", h) is True
    expanded = retrieval_query("When did he establish the CPP?", h)
    assert "first president" in expanded and "CPP" in expanded

    # Unrelated new topic
    assert is_followup_query("What is the capital of Ghana?", h) is False
    assert retrieval_query("What is the capital of Ghana?", h) == "What is the capital of Ghana?"
