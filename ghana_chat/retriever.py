"""Head-focused Knowledge Graph Retrieval Engine for Ghana Chat.

Matches user noun phrases and entities strictly against the HEAD of knowledge graph triples
(and not the tail). Prioritizes specific multi-word entities over generic single words.
"""

import os
import re
from typing import List, Dict, Any, Optional
import pandas as pd
from .abbreviations import AbbreviationResolver

STOPWORDS = {
    "what", "who", "where", "when", "why", "how", "tell", "me", "about",
    "is", "are", "was", "were", "be", "been", "being", "the", "a", "an",
    "in", "on", "at", "of", "for", "to", "and", "or", "do", "does", "did",
    "can", "you", "give", "please", "with", "from", "by", "that", "this",
    "these", "those", "have", "has", "had", "which", "its", "their", "his", "her"
}

# Generic words that often refer back to previously mentioned entities (anaphora)
GENERIC_ANAPHORA = {
    "organization", "organisation", "institution", "agency", "commission",
    "body", "company", "group", "person", "leader", "place", "area", "policy",
    "document", "plan", "programme", "program", "project", "thing", "service",
    "council", "ministry", "board", "committee"
}

_nlp = None


def get_spacy_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = False
    return _nlp


class HeadKGRetriever:
    """Fast in-memory index mapping lowercase head entities to knowledge triples."""

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            from . import config
            data_path = config.KG_PATH

        self.data_path = data_path
        self.heads_map: Dict[str, List[Dict[str, Any]]] = {}
        self.resolver = AbbreviationResolver()
        self.load_triples(data_path)

    def load_triples(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Knowledge Graph data file not found: {path}")

        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        h_low = df["head"].astype(str).str.strip().str.lower().values
        heads = df["head"].values
        rels = df["relation"].values
        tails = df["tail"].values
        ctrys = df["country"].values if "country" in df.columns else ["Ghana"] * len(df)
        sents = df["sentence"].values if "sentence" in df.columns else [""] * len(df)

        self.heads_map = {}
        count = 0
        for hl, h, r, t, c, s in zip(h_low, heads, rels, tails, ctrys, sents):
            if not hl or hl == "nan":
                continue
            if str(h).strip().lower() == str(t).strip().lower():
                continue
            if hl not in self.heads_map:
                self.heads_map[hl] = []
            self.heads_map[hl].append({
                "head": str(h).strip(),
                "relation": str(r).strip(),
                "tail": str(t).strip(),
                "country": str(c).strip(),
                "sentence": str(s).strip()
            })
            count += 1

        print(f"Indexed {count:,} triples across {len(self.heads_map):,} unique head entities from {path}")

    def extract_candidates(self, query: str) -> List[str]:
        """Extract candidate noun phrases, individual entities, and nouns from query text, augmented with resolved abbreviations."""
        candidates = []

        # 0. Resolve abbreviations / acronyms (e.g. NDPC -> National Development Planning Commission)
        pairs, aliases = self.resolver.resolve_query(query)
        for alias in aliases:
            if alias not in candidates:
                candidates.append(alias)

        nlp = get_spacy_nlp()

        if nlp:
            doc = nlp(query)

            # 1. Full noun chunks
            for chunk in doc.noun_chunks:
                words = [w for w in chunk.text.strip().lower().split() if w not in STOPWORDS]
                cleaned = " ".join(words)
                if len(cleaned) > 2 and cleaned not in candidates:
                    candidates.append(cleaned)

            # 2. Named entities
            for ent in doc.ents:
                words = [w for w in ent.text.strip().lower().split() if w not in STOPWORDS]
                cleaned = " ".join(words)
                if len(cleaned) > 2 and cleaned not in candidates:
                    candidates.append(cleaned)

            # 3. Individual NOUN and PROPN tokens (critical for decomposed concepts like 'cassava', 'farmers')
            for token in doc:
                t_low = token.text.strip().lower()
                if token.pos_ in ("NOUN", "PROPN") and t_low not in STOPWORDS and len(t_low) > 2:
                    if t_low not in candidates:
                        candidates.append(t_low)
                    lem = token.lemma_.strip().lower()
                    if lem not in STOPWORDS and len(lem) > 2 and lem not in candidates:
                        candidates.append(lem)

            # 4. Individual words from multi-word noun chunks
            for chunk in doc.noun_chunks:
                for w in chunk.text.strip().lower().split():
                    if w not in STOPWORDS and len(w) > 2 and w not in candidates:
                        candidates.append(w)
        else:
            # Fallback regex tokenization
            clean_q = re.sub(r"[^\w\s-]", " ", query).lower()
            words = [w for w in clean_q.split() if w not in STOPWORDS and len(w) > 2]
            for w in words:
                if w not in candidates:
                    candidates.append(w)
            for i in range(len(words)):
                for j in range(i + 1, min(i + 4, len(words) + 1)):
                    phrase = " ".join(words[i:j])
                    if phrase not in candidates:
                        candidates.append(phrase)

        # Sort longer, more specific phrases first, followed by individual nouns
        candidates.sort(key=lambda x: (-len(x.split()), -len(x)))
        return candidates

    def retrieve(self, query: str, max_triples: int = 10, country_filter: Optional[str] = "Ghana") -> Dict[str, Any]:
        """Retrieve triples where the candidate entity is strictly the HEAD."""
        candidates = self.extract_candidates(query)
        matched = []
        seen = set()

        # Primary pass: exact and multi-word candidate matches
        matched = []
        seen = set()
        per_cand_cap = max(3, min(5, max_triples // max(1, len(candidates))))

        for cand in candidates:
            cand_count = 0
            if cand in self.heads_map:
                for item in self.heads_map[cand]:
                    if country_filter and item["country"].lower() != country_filter.lower():
                        continue
                    t_key = (item["head"].lower(), item["relation"].lower(), item["tail"].lower())
                    if t_key not in seen:
                        seen.add(t_key)
                        matched.append(item)
                        cand_count += 1
                        if cand_count >= per_cand_cap or len(matched) >= max_triples:
                            break
            if len(matched) >= max_triples:
                break

        # Fallback without country filter if nothing found
        if not matched and country_filter:
            return self.retrieve(query, max_triples=max_triples, country_filter=None)

        return {
            "query": query,
            "entities": candidates,
            "triples": matched,
            "num_triples": len(matched)
        }

    def process_turn(
        self,
        query: str,
        history_entities: Optional[List[str]] = None,
        existing_triples: Optional[List[Dict[str, Any]]] = None,
        max_triples: int = 8,
        country_filter: Optional[str] = "Ghana"
    ) -> Dict[str, Any]:
        """Multi-turn retrieval processor.

        Checks if the query is a follow-up referencing already established context.
        - If NO new entities are introduced: returns is_followup=True, no new retrieval.
        - If NEW entities ARE introduced: retrieves head triples for the new entities
          and merges them with existing triples.
        """
        history_set = set(e.strip().lower() for e in (history_entities or []))
        existing_list = list(existing_triples or [])

        # Extract candidates from current query
        query_candidates = self.extract_candidates(query)

        # Filter out candidates that are purely generic anaphora referring back to prior context
        filtered_candidates = [
            c for c in query_candidates
            if c not in GENERIC_ANAPHORA and not (len(c.split()) == 1 and c in GENERIC_ANAPHORA)
        ]

        # Check for genuinely new entities not already in conversation history
        new_entities = []
        for c in filtered_candidates:
            # Check if this entity or a superstring was already processed
            already_known = (c in history_set) or any(c in h or h in c for h in history_set)
            if not already_known:
                new_entities.append(c)

        # Check for abbreviations in this query
        abbrev_pairs, _ = self.resolver.resolve_query(query)

        # Case A: Pure follow-up within existing context (no new entities introduced)
        if len(new_entities) == 0 and len(history_set) > 0:
            return {
                "query": query,
                "is_followup": True,
                "new_entities": [],
                "all_entities": list(history_set),
                "new_triples": [],
                "all_triples": existing_list,
                "active_triples": existing_list,
                "abbreviations": abbrev_pairs,
                "num_triples": len(existing_list)
            }

        # Case B: First turn or topic switch introducing new entities
        targets = new_entities if new_entities else query_candidates
        matched_new = []
        seen = set()
        per_cand_cap = max(3, max_triples // max(1, len(targets)))

        for cand in targets:
            cand_count = 0
            if cand in self.heads_map:
                for item in self.heads_map[cand]:
                    if country_filter and item["country"].lower() != country_filter.lower():
                        continue
                    t_key = (item["head"].lower(), item["relation"].lower(), item["tail"].lower())
                    if t_key not in seen:
                        seen.add(t_key)
                        matched_new.append(item)
                        cand_count += 1
                        if cand_count >= per_cand_cap or len(matched_new) >= max_triples:
                            break
            if len(matched_new) >= max_triples:
                break

        # Fallback if nothing matched and country filter was active
        if not matched_new and country_filter:
            return self.process_turn(
                query=query,
                history_entities=history_entities,
                existing_triples=existing_triples,
                max_triples=max_triples,
                country_filter=None
            )

        # Invalidate or augment with explicit definition triples for detected abbreviations
        definition_triples = []
        for abbr, full in abbrev_pairs:
            definition_triples.append({
                "head": abbr,
                "relation": "stands for",
                "tail": full,
                "country": "Ghana"
            })
            definition_triples.append({
                "head": full,
                "relation": "abbreviated as",
                "tail": abbr,
                "country": "Ghana"
            })

        updated_entities = list(history_set.union(set(targets)))
        active_triples = definition_triples + (matched_new if matched_new else [])
        all_triples = existing_list + definition_triples + matched_new

        return {
            "query": query,
            "is_followup": (len(history_set) > 0 and len(new_entities) == 0),
            "new_entities": new_entities if new_entities else targets,
            "all_entities": updated_entities,
            "new_triples": definition_triples + matched_new,
            "all_triples": all_triples,
            "active_triples": active_triples,
            "abbreviations": abbrev_pairs,
            "num_triples": len(active_triples)
        }
