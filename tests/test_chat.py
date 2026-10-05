"""Tests for Ghana Chat retrieval, multi-turn follow-up detection, and API endpoints."""

import pytest
from ghana_chat.retriever import HeadKGRetriever
from ghana_chat import config


def test_retriever_initialization():
    retriever = HeadKGRetriever(config.KG_PATH)
    assert len(retriever.heads_map) > 0


def test_head_focused_retrieval():
    retriever = HeadKGRetriever(config.KG_PATH)
    
    # Query about NDPC
    res = retriever.retrieve("Who is the chairperson of the NDPC?")
    triples = res.get("triples", [])
    assert len(triples) > 0
    heads = [t["head"].lower() for t in triples]
    assert any("ndpc" in h for h in heads)


def test_multiturn_followup_detection_no_new_entities():
    retriever = HeadKGRetriever(config.KG_PATH)
    
    # Turn 1: User asks about NDPC
    t1 = retriever.process_turn("Who is the chairperson of the NDPC?")
    assert t1["is_followup"] is False
    assert len(t1["new_triples"]) > 0
    assert len(t1["all_entities"]) > 0

    # Turn 2: User asks follow-up with pronoun ("Where is it located?")
    t2 = retriever.process_turn(
        query="Where is it located?",
        history_entities=t1["all_entities"],
        existing_triples=t1["all_triples"]
    )
    # Must detect as follow-up
    assert t2["is_followup"] is True
    # Must NOT perform new retrieval
    assert len(t2["new_triples"]) == 0
    # Must preserve previous triples
    assert len(t2["all_triples"]) == len(t1["all_triples"])


def test_multiturn_followup_with_new_entity():
    retriever = HeadKGRetriever(config.KG_PATH)
    
    # Turn 1
    t1 = retriever.process_turn("Who is the chairperson of the NDPC?")
    
    # Turn 2: Introduces "Ghana"
    t2 = retriever.process_turn(
        query="Does it operate in Ghana?",
        history_entities=t1["all_entities"],
        existing_triples=t1["all_triples"]
    )
    assert "ghana" in [e.lower() for e in t2["all_entities"]]
    # Should have merged new facts
    assert len(t2["all_triples"]) >= len(t1["all_triples"])


def test_no_reflexive_triples():
    retriever = HeadKGRetriever(config.KG_PATH)
    for head_key, items in retriever.heads_map.items():
        for t in items:
            assert t["head"].lower() != t["tail"].lower(), f"Reflexive triple found: {t}"
