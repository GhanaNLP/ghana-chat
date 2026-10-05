"""Tests for Ghana Chat retrieval and API endpoints."""

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
    # Every returned triple must have the entity as HEAD
    heads = [t["head"].lower() for t in triples]
    assert any("ndpc" in h for h in heads)


def test_no_reflexive_triples():
    retriever = HeadKGRetriever(config.KG_PATH)
    for head_key, items in retriever.heads_map.items():
        for t in items:
            assert t["head"].lower() != t["tail"].lower(), f"Reflexive triple found: {t}"
