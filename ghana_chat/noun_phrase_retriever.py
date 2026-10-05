"""Research Noun-Phrase Retrieval Engine for Ghana Chat.

Strict noun-phrase matching grounded solely on Ghanaian research theses (KNUST, UG, UCC, UEW, Ashesi):
  - Sourced from 420,135 academic research sentences (ghanaopenai/ghanaian-research-english-sentences).
  - Multi-word noun phrases, prepositional spans, and proper names are matched strictly as exact phrases.
  - No loose single-noun matching that causes false positives.
  - Abbreviation resolution (NDPC, CPP, EC, etc.) built in.
  - Zero embedding models, sub-millisecond memory-mapped postings lookup.
"""
from __future__ import annotations

import os
import re
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import pyarrow.parquet as pq

from .abbreviations import AbbreviationResolver

DEFAULT_KG_DIR = os.environ.get("GHANA_SENTENCE_INDEX", "/mnt/volume_d2wey28/projects/map-nav/kg_ghanaqa")

STOPWORDS = {
    "what", "who", "where", "when", "why", "how", "tell", "me", "about",
    "is", "are", "was", "were", "be", "been", "being", "the", "a", "an",
    "in", "on", "at", "of", "for", "to", "and", "or", "do", "does", "did",
    "can", "you", "give", "please", "with", "from", "by", "that", "this",
    "these", "those", "have", "has", "had"
}

QUESTION_TERMS = {
    "month", "year", "day", "week", "time", "date", "many", "much", "few",
    "people", "person", "man", "woman", "way", "thing", "things", "lot",
    "number", "amount", "part", "kind", "sort", "type", "name", "place",
    "area", "region", "country", "city", "town", "village", "home",
    "government", "state", "group", "level", "side", "end", "top", "bottom",
}

PRONOUNS_REF = {
    "he", "she", "it", "they", "him", "her", "them", "his", "hers", "their", "its",
    "this", "that", "these", "those", "the same"
}

GENERIC_ANAPHORA = {
    "he", "she", "it", "they", "him", "her", "them", "his", "hers", "their",
    "who", "what", "which", "where", "when", "how", "why", "someone", "anyone"
}

ATTRIBUTE_WORDS = {
    "wife", "husband", "children", "child", "son", "daughter", "family", "age",
    "born", "died", "education", "school", "career", "net worth", "salary",
    "role", "job", "position", "office", "party", "religion", "ethnicity"
}

DATE_PAT = re.compile(r"\b((?:19|20)\d\d)\b")
QUESTION_ECHO = re.compile(r"\?\s*$")


class SentenceNounPhraseRetriever:
    def __init__(self, index_dir: Optional[str] = None):
        self.index_dir = Path(index_dir or DEFAULT_KG_DIR)

        vocab_file = self.index_dir / "vocab.pkl"
        postings_file = self.index_dir / "postings.bin"
        sent_file = self.index_dir / "sentences.parquet"

        if not vocab_file.exists() or not postings_file.exists() or not sent_file.exists():
            raise FileNotFoundError(
                f"Index files missing in {self.index_dir}. Required: vocab.pkl, postings.bin, sentences.parquet"
            )

        with open(vocab_file, "rb") as f:
            self.vocab: Dict[str, Tuple[int, int]] = pickle.load(f)
        self.postings = np.memmap(postings_file, dtype=np.uint32, mode="r")

        tab = pq.read_table(sent_file)
        self.texts: List[str] = tab.column("text").to_pylist()
        self.sids: List[str] = tab.column("sid").to_pylist() if "sid" in tab.schema.names else [f"s_{i}" for i in range(len(self.texts))]
        col_doc = "doc_id" if "doc_id" in tab.schema.names else ("doc" if "doc" in tab.schema.names else None)
        self.doc_ids: List[Any] = tab.column(col_doc).to_pylist() if col_doc else list(range(len(self.texts)))
        self.sources: List[str] = tab.column("source").to_pylist() if "source" in tab.schema.names else ["ghanaqa"] * len(self.texts)
        self.titles: List[str] = tab.column("original_title").to_pylist() if "original_title" in tab.schema.names else [""] * len(self.texts)
        self.n: int = len(self.texts)

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

    def extract_noun_phrases(self, query: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Extract multi-word noun phrases, prepositional spans, and proper names."""
        nps: List[str] = []
        strong: List[str] = []

        # 0. Abbreviations (e.g. NDPC -> National Development Planning Commission)
        pairs, aliases = self.resolver.resolve_query(query)
        for alias in aliases:
            al = alias.strip().lower()
            if al not in nps:
                nps.append(al)
                strong.append(al)

        nlp = self._get_nlp()
        subject_np = None
        if nlp:
            doc = nlp(query)
            chunks = list(doc.noun_chunks)

            # 1. Identify the user query's subject noun phrase (e.g. "Kwame Nkrumah", "capital of Ghana")
            for i, c in enumerate(chunks):
                c_clean = re.sub(r"^(the|a|an|this|that|these|those|what|which)\s+", "", c.text.strip().lower()).strip()
                if c_clean in STOPWORDS or c_clean in QUESTION_TERMS or len(c_clean) <= 2:
                    continue
                if c.root.dep_ in ("nsubj", "nsubjpass"):
                    # Check for prepositional chaining: [noun] + (of/in/for) + [noun]
                    if i + 1 < len(chunks):
                        between = doc[c.end : chunks[i+1].start]
                        if len(between) == 1 and between[0].lower_ in ("of", "in", "for", "at"):
                            c2_clean = re.sub(r"^(the|a|an)\s+", "", chunks[i+1].text.strip().lower()).strip()
                            subject_np = f"{c_clean} {between[0].lower_} {c2_clean}"
                            break
                    subject_np = c_clean
                    break

            # If subject was an interrogative pronoun ("Who", "What"), identify the focal object
            if not subject_np:
                for c in chunks:
                    c_clean = re.sub(r"^(the|a|an|this|that|these|those|what|which)\s+", "", c.text.strip().lower()).strip()
                    if c_clean in STOPWORDS or c_clean in QUESTION_TERMS or len(c_clean) <= 2:
                        continue
                    if c.root.dep_ in ("dobj", "pobj", "attr"):
                        subject_np = c_clean
                        break

            if subject_np and subject_np not in nps:
                nps.insert(0, subject_np)
                strong.insert(0, subject_np)

            # 2. Prepositional phrases: e.g. "capital of Ghana", "chairperson of the NDPC"
            for i in range(len(doc) - 2):
                if doc[i].pos_ in ("NOUN", "PROPN") and doc[i + 1].lower_ in ("of", "in", "for", "at"):
                    span = doc[i : min(len(doc), i + 6)]
                    p = " ".join(t.text for t in span if t.pos_ != "PUNCT").strip().lower()
                    p = re.sub(r"^(the|a|an|this|that|these|those)\s+", "", p).strip()
                    if len(p.split()) >= 2 and p not in nps:
                        nps.append(p)
                        strong.append(p)

            # 3. Named entities (PERSON, ORG, GPE, LOC)
            for ent in doc.ents:
                if ent.label_ in ("DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"):
                    continue
                words = [w for w in ent.text.strip().lower().split() if w not in STOPWORDS]
                cleaned = " ".join(words).strip()
                if len(cleaned) > 2 and cleaned not in nps:
                    nps.append(cleaned)
                    strong.append(cleaned)

            # 4. Noun chunks
            for chunk in doc.noun_chunks:
                c = chunk.text.strip().lower()
                c_clean = re.sub(r"^(the|a|an|this|that|these|those|what|which)\s+", "", c).strip()
                words = c_clean.split()
                if len(words) >= 2 and c_clean not in nps:
                    if not all(w in QUESTION_TERMS for w in words):
                        nps.append(c_clean)
                        strong.append(c_clean)
                elif len(words) == 1 and chunk.root.pos_ in ("NOUN", "PROPN") and c_clean not in nps:
                    if c_clean not in STOPWORDS and c_clean not in QUESTION_TERMS:
                        nps.append(c_clean)

                # Head noun of the chunk (e.g. "cassava" in "farmers plant cassava")
                root = chunk.root.text.strip().lower()
                if root not in STOPWORDS and root not in QUESTION_TERMS and len(root) > 2 and root not in nps:
                    nps.append(root)

            # 5. Fallback to key nouns only if no multi-word phrase was captured
            if not nps:
                for token in doc:
                    if token.pos_ in ("NOUN", "PROPN"):
                        t_low = token.text.strip().lower()
                        if t_low not in STOPWORDS and t_low not in QUESTION_TERMS and len(t_low) > 2:
                            nps.append(t_low)
        else:
            clean_q = re.sub(r"[^\w\s-]", " ", query).lower()
            words = [w for w in clean_q.split() if w not in STOPWORDS and len(w) > 2 and w not in QUESTION_TERMS]
            for w in words:
                if w not in nps:
                    nps.append(w)
            for i in range(len(words)):
                for j in range(i + 1, min(i + 4, len(words) + 1)):
                    phrase = " ".join(words[i:j])
                    if phrase not in nps:
                        nps.append(phrase)

        # Sort longer, more specific phrases first
        rank = {c: i for i, c in enumerate(strong)}
        nps.sort(key=lambda x: (-len(x.split()), -len(x), rank.get(x, 10**6)))

        # Prune fragments that are strict sub-phrases of a longer phrase
        pruned: List[str] = []
        for c in nps:
            if any(c != other and f" {c} " in f" {other} " for other in nps):
                continue
            pruned.append(c)

        return (pruned if pruned else nps), pairs

    def search_exact_phrase(self, phrase: str, max_hits: int = 2000) -> List[int]:
        """Look up all sentence IDs where the exact noun phrase occurs."""
        tokens = [w for w in re.findall(r"[a-zA-Z0-9]+", phrase.lower()) if w in self.vocab]
        if not tokens:
            return []

        tokens.sort(key=lambda w: self.vocab[w][1])
        off, cnt = self.vocab[tokens[0]]
        sids = set(self.postings[off : off + cnt])

        for t in tokens[1:]:
            off_t, cnt_t = self.vocab[t]
            sids.intersection_update(self.postings[off_t : off_t + cnt_t])
            if not sids:
                return []

        phrase_l = phrase.lower()
        matched = []
        for sid in sids:
            if phrase_l in self.texts[sid].lower():
                matched.append(sid)
                if len(matched) >= max_hits:
                    break
        return matched

    def score_sentence(self, text: str, phrase: str, query_words: Set[str], all_nps: Optional[List[str]] = None) -> float:
        score = 10.0
        text_l = text.lower()
        phrase_l = phrase.lower()

        # Penalize question echoes
        if QUESTION_ECHO.search(text):
            score -= 15.0

        # Exact phrase position
        idx = text_l.find(phrase_l)
        if idx == 0:
            score += 5.0
        elif 0 < idx < 40:
            score += 3.0

        # Multi-Noun-Phrase Co-occurrence:
        # Heavily prioritize sentences that contain multiple/all extracted query noun phrases!
        if all_nps:
            np_matches = sum(1 for p in all_nps if p in text_l)
            if np_matches > 1:
                score += (np_matches - 1) * 25.0

        # Clean sentence length
        L = len(text)
        if 60 <= L <= 280:
            score += 3.0
        elif L < 35:
            score -= 5.0

        # Definitional / action verbs
        if any(w in text_l for w in [" is ", " was ", " serves as ", " established ", " founded ", " appointed ", " operates "]):
            score += 2.5

        # Query word overlap
        extra = sum(1 for w in query_words if w in text_l and w not in phrase_l)
        score += extra * 3.0

        # Recency boost
        dates = [int(m) for m in DATE_PAT.findall(text)]
        if dates:
            score += 2.0
            if max(dates) >= 2020:
                score += 3.0

        return score

    def is_subject_phrase(self, doc, phrase: str) -> int:
        """Returns 1 if phrase acts as grammatical subject, subject appositive, or predicate attribute."""
        if doc is None:
            return 0
        phrase_words = set(phrase.lower().split())
        for token in doc:
            if token.text.lower() in phrase_words:
                if token.dep_ in ("nsubj", "nsubjpass"):
                    return 1
                if token.dep_ == "appos" and token.head.dep_ in ("nsubj", "nsubjpass"):
                    return 1
                if token.dep_ == "attr" and token.head.pos_ in ("AUX", "VERB"):
                    return 1
        return 0

    def retrieve(
        self,
        query: str,
        max_sources: int = 4,
        candidates: Optional[List[str]] = None,
        neighbour_window: int = 1
    ) -> Dict[str, Any]:
        """Retrieve relevant source sentences, strictly prioritizing multi-noun-phrase co-occurrence."""
        if candidates is None:
            nps, abbrevs = self.extract_noun_phrases(query)
        else:
            nps = candidates
            _, abbrevs = self.resolver.resolve_query(query)

        query_words = set(re.findall(r"\b[a-zA-Z0-9'-]+\b", query.lower())) - STOPWORDS

        if not nps:
            return {"query": query, "candidates": [], "abbreviations": abbrevs, "sources": [], "num_sources": 0, "grounded": False}

        # 1. Search sentence matches for each noun phrase
        phrase_hits: Dict[str, List[int]] = {}
        sentence_matching_nps: Dict[int, List[str]] = defaultdict(list)

        for phrase in nps:
            matched = self.search_exact_phrase(phrase)
            phrase_hits[phrase] = matched
            for sid in matched:
                sentence_matching_nps[sid].append(phrase)

        if not sentence_matching_nps:
            return {"query": query, "candidates": nps, "abbreviations": abbrevs, "sources": [], "num_sources": 0, "grounded": False}

        # 2. Group candidate sentences by number of matched noun phrases (descending: N, N-1, ..., 1)
        grouped_by_count: Dict[int, List[Tuple[int, List[str]]]] = defaultdict(list)
        for sid, matched_phrases in sentence_matching_nps.items():
            grouped_by_count[len(matched_phrases)].append((sid, matched_phrases))

        all_ranked: List[Tuple[int, Tuple[int, int, int, int], str]] = []

        # 3. Fill budget from highest overlap count down to 1:
        # 1st criteria: noun phrases that occur (more noun phrases = higher priority)
        # 2nd criteria: noun phrase is the SUBJECT (is_subject = 1 before 0)
        # 3rd criteria: length of the text (more text preferred)
        # 4th criteria: date (more current/recent year the better)
        nlp = self._get_nlp()
        for count in sorted(grouped_by_count.keys(), reverse=True):
            items = grouped_by_count[count]
            # Filter question echoes first
            valid_items = [(sid, m_nps) for sid, m_nps in items if not QUESTION_ECHO.search(self.texts[sid])]
            if not valid_items:
                continue

            # Pre-filter up to 100 candidates to evaluate subject dependency parsing cleanly
            valid_items.sort(key=lambda x: len(self.texts[x[0]]), reverse=True)
            eval_items = valid_items[:100]

            docs = list(nlp.pipe([self.texts[s] for s, _ in eval_items], batch_size=50)) if nlp else [None] * len(eval_items)

            scored_tier = []
            for (sid, matched_phrases), doc in zip(eval_items, docs):
                t = self.texts[sid]
                np_count = len(matched_phrases)
                is_subj = max((self.is_subject_phrase(doc, p) for p in matched_phrases), default=0)
                text_len = min(len(t.strip()), 450)
                dates = [int(y) for y in DATE_PAT.findall(t) if 1950 <= int(y) <= 2026]
                max_year = max(dates) if dates else 0

                ranking_key = (np_count, is_subj, text_len, max_year)
                scored_tier.append((sid, ranking_key, " + ".join(matched_phrases)))

            # Sort descending by (np_count, is_subject, text_len, date)
            scored_tier.sort(key=lambda x: x[1], reverse=True)
            all_ranked.extend(scored_tier)

        # Document Diversity Selection:
        # Prioritize sentences from different articles; only take multiple sentences
        # from the same article if there aren't enough distinct matching articles.
        seen_sids = set()
        seen_docs = set()
        chosen: List[Tuple[int, float, str]] = []

        # Pass A: 1 sentence per article
        for sid, rkey, p_name in all_ranked:
            d = self.doc_ids[sid]
            if d not in seen_docs:
                seen_docs.add(d)
                seen_sids.add(sid)
                display_score = round(
                    rkey[0] * 10.0 + rkey[1] * 5.0 + rkey[2] / 100.0 + ((rkey[3] - 2000) * 0.1 if rkey[3] else 0.0),
                    2
                )
                chosen.append((sid, display_score, p_name))
                if len(chosen) >= max_sources:
                    break

        # Pass B: If fewer than max_sources distinct articles were found, fill from remaining sentences
        if len(chosen) < max_sources:
            for sid, rkey, p_name in all_ranked:
                if sid not in seen_sids:
                    seen_sids.add(sid)
                    display_score = round(
                        rkey[0] * 10.0 + rkey[1] * 5.0 + rkey[2] / 100.0 + ((rkey[3] - 2000) * 0.1 if rkey[3] else 0.0),
                        2
                    )
                    chosen.append((sid, display_score, p_name))
                    if len(chosen) >= max_sources:
                        break

        # Near-duplicate suppression (Jaccard token overlap > 0.8)
        filtered_chosen = []
        seen_tokens = []
        for sid, sc, phrase in chosen:
            toks = set(re.findall(r"[a-z0-9]+", self.texts[sid].lower()))
            if any(len(toks & t) / max(1, len(toks | t)) > 0.8 for t in seen_tokens):
                continue
            seen_tokens.append(toks)
            filtered_chosen.append((sid, sc, phrase))

        results = []
        for sid, score, phrase in filtered_chosen:
            lo = max(0, sid - neighbour_window)
            hi = min(self.n, sid + neighbour_window + 1)
            ctx = " ".join(self.texts[lo:hi]).strip() if neighbour_window > 0 else self.texts[sid]

            results.append({
                "sid": self.sids[sid],
                "doc_id": self.doc_ids[sid],
                "sentence": self.texts[sid],
                "context": ctx,
                "score": round(score, 2),
                "matched_candidate": phrase,
                "source": self.sources[sid],
                "title": self.titles[sid]
            })

        return {
            "query": query,
            "candidates": nps,
            "abbreviations": abbrevs,
            "sources": results,
            "num_sources": len(results),
            "grounded": len(results) > 0 and any(r["score"] > 8.0 for r in results)
        }

    def process_turn(
        self,
        query: str,
        history_entities: Optional[List[str]] = None,
        existing_sources: Optional[List[Dict[str, Any]]] = None,
        max_sources: int = 4
    ) -> Dict[str, Any]:
        """Multi-turn retrieval with referential pronoun tracking and topic-switch management."""
        history_set = set(e.strip().lower() for e in (history_entities or []))
        query_words = set(re.findall(r"\b[a-zA-Z0-9'-]+\b", query.lower()))
        has_referential_pronoun = bool(query_words & PRONOUNS_REF) and len(history_set) > 0

        nps, abbrevs = self.extract_noun_phrases(query)

        filtered_candidates = []
        for c in nps:
            c_low = c.lower()
            if c_low in GENERIC_ANAPHORA:
                continue
            if has_referential_pronoun and (c_low in ATTRIBUTE_WORDS or any(w in ATTRIBUTE_WORDS for w in c_low.split())):
                continue
            filtered_candidates.append(c)

        new_entities = []
        for c in filtered_candidates:
            already_known = (c in history_set) or any(c in h or h in c for h in history_set)
            if not already_known:
                new_entities.append(c)

        is_followup = (len(history_set) > 0 and len(new_entities) == 0) or has_referential_pronoun

        if is_followup:
            target_candidates = list(history_set) + filtered_candidates
        else:
            target_candidates = filtered_candidates if filtered_candidates else nps

        ret = self.retrieve(query, max_sources=max_sources, candidates=target_candidates)
        updated_entities = list(history_set.union(set(target_candidates)))

        return {
            "query": query,
            "is_followup": is_followup,
            "candidates": target_candidates,
            "new_entities": new_entities,
            "all_entities": updated_entities,
            "sources": ret["sources"],
            "num_sources": ret["num_sources"],
            "grounded": ret["grounded"],
            "abbreviations": abbrevs
        }
