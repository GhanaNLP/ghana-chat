"""Verbalized Knowledge Graph Retrieval Engine for Ghana Chat.

Strict Exact Noun-Phrase Matching with NO Fallback:
  - Indexes 58,034 clean, non-repetitive knowledge graph facts + injected abbreviation assertions.
  - Matches candidate noun phrases strictly on exact entities (no arbitrary semantic embedding fallback).
  - Prepositional titles and full names ("Coordinated Programme for Social and Economic Development") are treated as single unified noun phrases.
  - Bumps max retrieved facts capacity up to 300.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import pyarrow.parquet as pq

from .abbreviations import AbbreviationResolver

DEFAULT_KG_DIR = os.environ.get("GHANA_VERBALIZED_KG_DIR", "/mnt/volume_d2wey28/projects/ghana-chat/data/verbalized_kg")

TITLE_CONNECTORS = {"of", "for", "in", "on", "at", "and", "&", "to", "the"}
LEAD_QUESTION = re.compile(r"^(what|who|which|where|when|why|how|tell\s+me\s+about|explain)\s+(is|are|was|were|do|does|did|the|a|an)?\s*", re.I)
CLEAN_DET = re.compile(r"^(the|a|an)\s+", re.I)

STOPWORDS = {
    "what", "who", "where", "when", "why", "how", "tell", "me", "about",
    "is", "are", "was", "were", "be", "been", "being", "the", "a", "an",
    "in", "on", "at", "of", "for", "to", "and", "or", "do", "does", "did",
    "can", "you", "give", "please", "with", "from", "by", "that", "this",
    "these", "those", "have", "has", "had"
}

PRONOUNS_REF = {
    "he", "she", "it", "they", "him", "her", "them", "his", "hers", "their", "its",
    "this", "that", "these", "those", "the same"
}

GENERIC_ANAPHORA = {
    "he", "she", "it", "they", "him", "her", "them", "his", "hers", "their",
    "who", "what", "which", "where", "when", "how", "why", "someone", "anyone"
}


def verbalize(h: str, r: str, t: str) -> str:
    h, t = h.strip(), t.strip()
    rl = r.strip().lower()
    if rl == "instance of":
        art = "an" if t.lower()[:1] in "aeiou" else "a"
        return f"{h} is {art} {t}."
    elif rl == "subclass of":
        return f"{h} is a type of {t}."
    elif rl == "country":
        return f"{h} is in {t}."
    elif rl == "located in":
        return f"{h} is located in {t}."
    elif rl == "part of":
        return f"{h} is part of {t}."
    elif rl == "has part":
        return f"{h} includes {t}."
    elif rl == "employer":
        return f"{h} employs {t}."
    elif rl == "member of":
        return f"{h} is a member of {t}."
    elif rl == "participant in":
        return f"{h} participated in {t}."
    elif rl == "creator":
        return f"{h} created {t}."
    elif rl == "author":
        return f"{h} is the author of {t}."
    elif rl == "owned by":
        return f"{h} is owned by {t}."
    elif rl == "subsidiary":
        return f"{h} is a subsidiary of {t}."
    elif rl == "inception":
        return f"{h} was established in {t}."
    elif rl == "part of a series":
        return f"{h} is part of the series {t}."
    elif rl == "abbreviation for":
        return f"{h} is an abbreviation for {t}."
    else:
        rc = r.replace("_", " ").strip()
        return f"{h} {rc} {t}."


class VerbalizedKGRetriever:
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir or DEFAULT_KG_DIR)

        sent_file = self.data_dir / "sentences.parquet"
        if not sent_file.exists():
            raise FileNotFoundError(f"Verbalized KG file missing: {sent_file}")

        tab = pq.read_table(sent_file)
        self.ids = tab.column("id").to_pylist()
        self.texts = tab.column("text").to_pylist()
        self.heads = tab.column("head").to_pylist()
        self.relations = tab.column("relation").to_pylist()
        self.tails = tab.column("tail").to_pylist()
        self.categories = tab.column("category").to_pylist()
        self.n = len(self.texts)

        # Build exact normalized entity index (head & tail -> list of indices)
        self.entity_map: Dict[str, List[int]] = {}
        for idx in range(self.n):
            hl = CLEAN_DET.sub("", self.heads[idx]).strip().lower()
            tl = CLEAN_DET.sub("", self.tails[idx]).strip().lower()
            if hl:
                self.entity_map.setdefault(hl, []).append(idx)
            if tl and tl != hl:
                self.entity_map.setdefault(tl, []).append(idx)

        self.resolver = AbbreviationResolver()
        self._nlp = None

    def _get_nlp(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = False
        return self._nlp

    def extract_exact_noun_phrases(self, query: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Extract exact unified proper noun phrases, titles, and abbreviation aliases."""
        abbrev_pairs, aliases = self.resolver.resolve_query(query)
        candidates: List[str] = list(aliases)

        clean_q = query.strip().rstrip("?!.")
        core = LEAD_QUESTION.sub("", clean_q).strip()
        core_words = core.split()

        # Check if entire core phrase is a unified title entity (Capitalized words + connectors)
        if len(core_words) >= 2:
            is_title = all(w[0].isupper() or w.lower() in TITLE_CONNECTORS for w in core_words if w)
            if is_title:
                clean_title = CLEAN_DET.sub("", core).strip().lower()
                if clean_title not in candidates:
                    candidates.insert(0, clean_title)

        if not candidates:
            nlp = self._get_nlp()
            if nlp:
                doc = nlp(clean_q)
                chunks = list(doc.noun_chunks)
                i = 0
                while i < len(chunks):
                    c1 = chunks[i]
                    c1_t = CLEAN_DET.sub("", c1.text).strip().lower()
                    if i + 1 < len(chunks):
                        c2 = chunks[i + 1]
                        between = doc[c1.end : c2.start]
                        if all(t.lower_ in TITLE_CONNECTORS for t in between):
                            c2_t = CLEAN_DET.sub("", c2.text).strip().lower()
                            mid = " ".join(t.text.lower() for t in between)
                            full = f"{c1_t} {mid} {c2_t}".strip()
                            if full not in candidates:
                                candidates.append(full)
                            i += 2
                            continue
                    if len(c1_t) > 2 and c1_t not in ("what", "who", "which", "how"):
                        if c1_t not in candidates:
                            candidates.append(c1_t)
                    i += 1

                for ent in doc.ents:
                    if ent.label_ not in ("DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "QUANTITY"):
                        et = CLEAN_DET.sub("", ent.text).strip().lower()
                        if len(et) > 2 and et not in candidates:
                            candidates.append(et)
            else:
                clean_q = re.sub(r"[^\w\s-]", " ", query).lower()
                words = [w for w in clean_q.split() if w not in STOPWORDS and len(w) > 2]
                for w in words:
                    if w not in candidates:
                        candidates.append(w)

        # Check bidirectional abbreviation aliases for EVERY candidate noun phrase
        expanded_candidates: List[str] = []
        resolved_abbrevs: List[Tuple[str, str]] = list(abbrev_pairs)

        for p in candidates:
            pl = p.lower().strip()
            if pl not in expanded_candidates:
                expanded_candidates.append(pl)

            # 1. If p matches an acronym (e.g. "ndc" -> "NDC" -> "National Democratic Congress")
            if pl.upper() in self.resolver.abbrev_map:
                full = self.resolver.abbrev_map[pl.upper()]
                pair = (pl.upper(), full)
                if pair not in resolved_abbrevs:
                    resolved_abbrevs.append(pair)
                fl = full.lower()
                combo = f"{fl} ({pl})"
                for alias in [fl, combo]:
                    if alias not in expanded_candidates:
                        expanded_candidates.append(alias)

            # 2. If p matches a full name (e.g. "national democratic congress" -> "NDC")
            if pl in self.resolver.reverse_map:
                abbr = self.resolver.reverse_map[pl]
                full = self.resolver.abbrev_map[abbr]
                pair = (abbr, full)
                if pair not in resolved_abbrevs:
                    resolved_abbrevs.append(pair)
                al = abbr.lower()
                combo = f"{pl} ({al})"
                for alias in [al, combo]:
                    if alias not in expanded_candidates:
                        expanded_candidates.append(alias)

        return expanded_candidates, resolved_abbrevs

    def retrieve(self, query: str, top_k: int = 300) -> Dict[str, Any]:
        """Strict exact noun-phrase matching with NO random fallback. Max capacity up to 300 facts."""
        candidates, abbrev_pairs = self.extract_exact_noun_phrases(query)

        seen_pairs = set()
        results = []

        # REQUIREMENT: Where abbreviations are involved, include as FIRST item:
        # "xxx is the abbreviation of xxx" for the LLM to have in context
        if abbrev_pairs:
            for abbr, full in abbrev_pairs:
                abbrev_sent = f"{abbr} is the abbreviation of {full}."
                pair_key = frozenset({abbr.lower(), full.lower()})
                seen_pairs.add(pair_key)
                results.append({
                    "sid": 999900,
                    "doc_id": "kg_abbrev",
                    "sentence": abbrev_sent,
                    "context": abbrev_sent,
                    "head": abbr,
                    "relation": "abbreviation of",
                    "tail": full,
                    "score": 100.0,
                    "matched_candidate": abbr
                })

        for cand in candidates:
            cand_l = cand.strip().lower()
            if not cand_l:
                continue

            # 1. Exact entity lookup in entity_map
            matches = self.entity_map.get(cand_l, [])
            for idx in matches:
                h = self.heads[idx]
                t = self.tails[idx]
                pair_key = frozenset({h.lower(), t.lower()})
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    results.append({
                        "sid": self.ids[idx],
                        "doc_id": f"kg_{self.ids[idx]}",
                        "sentence": self.texts[idx],
                        "context": self.texts[idx],
                        "head": h,
                        "relation": self.relations[idx],
                        "tail": t,
                        "score": 10.0,
                        "matched_candidate": cand
                    })
                    if len(results) >= top_k:
                        break

            # 2. Whole-word / whole-phrase containment ONLY (never partial sub-token overlap)
            if len(results) < top_k:
                cand_re = re.compile(r"\b" + re.escape(cand_l) + r"\b")
                for ent_key, idx_list in self.entity_map.items():
                    if cand_re.search(ent_key):
                        for idx in idx_list:
                            h = self.heads[idx]
                            t = self.tails[idx]
                            pair_key = frozenset({h.lower(), t.lower()})
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)
                                results.append({
                                    "sid": self.ids[idx],
                                    "doc_id": f"kg_{self.ids[idx]}",
                                    "sentence": self.texts[idx],
                                    "context": self.texts[idx],
                                    "head": h,
                                    "relation": self.relations[idx],
                                    "tail": t,
                                    "score": 8.0,
                                    "matched_candidate": cand
                                })
                                if len(results) >= top_k:
                                    break
                    if len(results) >= top_k:
                        break

            if len(results) >= top_k:
                break

        return {
            "query": query,
            "candidates": candidates,
            "abbreviations": abbrev_pairs,
            "sources": results,
            "triples": [{"head": r["head"], "relation": r["relation"], "tail": r["tail"]} for r in results],
            "num_sources": len(results),
            "grounded": len(results) > 0
        }

    def process_turn(
        self,
        query: str,
        history_entities: Optional[List[str]] = None,
        existing_sources: Optional[List[Dict[str, Any]]] = None,
        max_sources: int = 300
    ) -> Dict[str, Any]:
        """Multi-turn retrieval with anaphora expansion and strict exact matching."""
        history_set = set(e.strip().lower() for e in (history_entities or []))
        query_words = set(re.findall(r"\b[a-zA-Z0-9'-]+\b", query.lower()))
        has_referential_pronoun = bool(query_words & PRONOUNS_REF) and len(history_set) > 0

        candidates, abbrev_pairs = self.extract_exact_noun_phrases(query)
        is_followup = (len(history_set) > 0 and len(candidates) == 0) or has_referential_pronoun

        # Expand retrieval query with previous context if pronoun follow-up
        eff_query = query
        if is_followup and history_entities:
            eff_query = f"{history_entities[-1]} {query}"

        ret = self.retrieve(eff_query, top_k=max_sources)
        updated_entities = list(history_set.union(set(candidates)))

        return {
            "query": query,
            "is_followup": is_followup,
            "candidates": candidates,
            "all_entities": updated_entities,
            "sources": ret["sources"],
            "triples": ret["triples"],
            "num_sources": ret["num_sources"],
            "grounded": ret["grounded"],
            "abbreviations": abbrev_pairs
        }
