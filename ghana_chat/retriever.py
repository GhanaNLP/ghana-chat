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

# Words that refer to an established subject in context
PRONOUNS_REF = {
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "this", "that"
}

# Attribute / property queries that describe a subject rather than a new standalone topic
ATTRIBUTE_WORDS = {
    "children", "child", "wife", "husband", "spouse", "son", "daughter", "family",
    "age", "birthday", "birth", "born", "death", "died", "hometown", "parents",
    "father", "mother", "school", "education", "religion", "salary", "career",
    "qualifications", "degree", "net worth", "background"
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
        # entity -> triples where it appears as head OR tail. The GraphQA approach
        # scans all triples for the entity, which avoids the head-only blind spot
        # (e.g. "IOS developer apple" is invisible to an "About Apple" question).
        self.any_map: Dict[str, List[Dict[str, Any]]] = {}
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
        self.any_map = {}
        count = 0
        for hl, h, r, t, c, s in zip(h_low, heads, rels, tails, ctrys, sents):
            if not hl or hl == "nan":
                continue
            if str(h).strip().lower() == str(t).strip().lower():
                continue
            item = {
                "head": str(h).strip(),
                "relation": str(r).strip(),
                "tail": str(t).strip(),
                "country": str(c).strip(),
                "sentence": str(s).strip()
            }
            self.heads_map.setdefault(hl, []).append(item)

            # Index the same triple under the tail's key in any_map so that
            # entity-in-tail questions are answerable.
            tl = str(t).strip().lower()
            self.any_map.setdefault(hl, []).append(item)
            if tl and tl != "nan" and tl != hl:
                self.any_map.setdefault(tl, []).append(item)
            count += 1

        print(f"Indexed {count:,} triples across {len(self.heads_map):,} unique head entities from {path}")

    @staticmethod
    def _is_noise(triple: Dict[str, Any]) -> bool:
        """True if the triple encodes ontology scaffolding rather than a real fact."""
        from .generator import normalize_relation_key
        from . import config
        return normalize_relation_key(triple.get("relation", "")) in config.ONTOLOGY_NOISE_KEYS

    def extract_candidates(self, query: str) -> List[str]:
        """Extract candidate entity names from a query.

        Proper-noun spans (spaCy NER) are kept intact so multi-word names such as
        "Asiedu Nketia" are never fragmented into single tokens that could match a
        *different* entity. Generic question terms are dropped because they
        describe the question rather than an entity in the graph.
        """
        from . import config

        candidates: List[str] = []
        strong: List[str] = []

        # 0. Resolve abbreviations / acronyms (e.g. NDPC -> National Development Planning Commission)
        pairs, aliases = self.resolver.resolve_query(query)
        for alias in aliases:
            if alias not in candidates:
                candidates.append(alias)
                strong.append(alias)

        nlp = get_spacy_nlp()

        if nlp:
            doc = nlp(query)

            # 1. Named entities -- highest-signal candidates. Keep the full span.
            proper_spans = []
            for ent in doc.ents:
                if ent.label_ in ("DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"):
                    continue
                words = [w for w in ent.text.strip().lower().split() if w not in STOPWORDS]
                cleaned = " ".join(words).strip()
                if len(cleaned) > 2 and cleaned not in candidates:
                    candidates.append(cleaned)
                    strong.append(cleaned)
                    proper_spans.append(cleaned)

            # 2. Full noun chunks for concept queries (e.g. "growing cocoa").
            for chunk in doc.noun_chunks:
                words = [w for w in chunk.text.strip().lower().split() if w not in STOPWORDS]
                cleaned = " ".join(words).strip()
                if len(cleaned) > 2 and cleaned not in candidates:
                    if not any(w in config.QUESTION_TERM_STOPLIST for w in cleaned.split()):
                        candidates.append(cleaned)
                        strong.append(cleaned)

            # 3. Individual common nouns so decomposed concepts still match
            #    (e.g. "cassava" in "how do farmers grow cassava").
            weak = []
            for token in doc:
                if token.pos_ != "NOUN":
                    continue
                t_low = token.text.strip().lower()
                if t_low in STOPWORDS or len(t_low) <= 2:
                    continue
                if t_low in config.QUESTION_TERM_STOPLIST or t_low in candidates:
                    continue
                candidates.append(t_low)
                weak.append(t_low)
                lem = token.lemma_.strip().lower()
                if (
                    lem not in STOPWORDS and len(lem) > 2
                    and lem not in candidates
                    and lem not in config.QUESTION_TERM_STOPLIST
                ):
                    candidates.append(lem)
                    weak.append(lem)

            # 4. Drop bare tokens that are fragments of a proper-noun span we already
            #    captured. This is what previously let "Asiedu" match a different person.
            for span in proper_spans:
                for tok in span.split():
                    if tok != span and tok in candidates:
                        candidates.remove(tok)
                        if tok in weak:
                            weak.remove(tok)
        else:
            # Fallback regex tokenization
            clean_q = re.sub(r"[^\w\s-]", " ", query).lower()
            words = [w for w in clean_q.split() if w not in STOPWORDS and len(w) > 2
                     and w not in config.QUESTION_TERM_STOPLIST]
            for w in words:
                if w not in candidates:
                    candidates.append(w)
                    strong.append(w)
            for i in range(len(words)):
                for j in range(i + 1, min(i + 4, len(words) + 1)):
                    phrase = " ".join(words[i:j])
                    if phrase not in candidates:
                        candidates.append(phrase)
                        strong.append(phrase)

        # Sort longer, more specific phrases first, then by extraction strength.
        rank = {c: i for i, c in enumerate(strong)}
        candidates.sort(key=lambda x: (-len(x.split()), -len(x), rank.get(x, 10**6)))
        return candidates

    def retrieve(
        self,
        query: str,
        max_triples: int = 10,
        country_filter: Optional[str] = "Ghana",
        candidates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Retrieve triples relevant to the query's entities.

        Pass 1 (preferred): entity is the triple HEAD -- these give the cleanest,
        subject-centric facts. Pass 2 (fallback): entity appears as the TAIL, so
        inbound relationships ("... developed Apple") are not lost.
        """
        if candidates is None:
            candidates = self.extract_candidates(query)

        seen = set()
        matched: List[Dict[str, Any]] = []

        def _acceptable(item: Dict[str, Any]) -> bool:
            if country_filter and str(item.get("country", "")).lower() != country_filter.lower():
                return False
            if self._is_noise(item):
                return False
            t_key = (item["head"].lower(), item["relation"].lower(), item["tail"].lower())
            if t_key in seen:
                return False
            seen.add(t_key)
            return True

        # Pass 1: head matches, most specific candidates first.
        per_cand_cap = max(3, min(5, max_triples // max(1, len(candidates))))
        for cand in candidates:
            if len(matched) >= max_triples:
                break
            added = 0
            for item in self.heads_map.get(cand, []):
                if not _acceptable(item):
                    continue
                matched.append(item)
                added += 1
                if added >= per_cand_cap or len(matched) >= max_triples:
                    break

        # Pass 2: entity appears as tail (only when the head pass is thin).
        if len(matched) < max_triples:
            for cand in candidates:
                if len(matched) >= max_triples:
                    break
                for item in self.any_map.get(cand, []):
                    if str(item["head"]).strip().lower() == cand:
                        continue  # already covered by the head pass
                    if not _acceptable(item):
                        continue
                    matched.append(item)
                    if len(matched) >= max_triples:
                        break

        # Fallback without country filter if nothing found
        if not matched and country_filter:
            return self.retrieve(query, max_triples=max_triples, country_filter=None, candidates=candidates)

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

        # Extract tokens and candidate entities from query
        query_words = set(re.findall(r"\b[a-zA-Z0-9'-]+\b", query.lower()))
        has_referential_pronoun = bool(query_words & PRONOUNS_REF) and len(history_set) > 0

        # Extract candidates from current query
        query_candidates = self.extract_candidates(query)

        # Filter out candidates that are purely generic anaphora or attribute queries when referencing prior entity
        filtered_candidates = []
        for c in query_candidates:
            c_low = c.lower()
            if c_low in GENERIC_ANAPHORA or (len(c_low.split()) == 1 and c_low in GENERIC_ANAPHORA):
                continue
            # If the user used a pronoun ("he", "she", "it"), attribute words (e.g. "children", "wife")
            # are properties of the subject, not standalone topic entities
            if has_referential_pronoun and (c_low in ATTRIBUTE_WORDS or any(w in ATTRIBUTE_WORDS for w in c_low.split())):
                continue
            filtered_candidates.append(c)

        # Check for genuinely new entities not already in conversation history
        new_entities = []
        for c in filtered_candidates:
            already_known = (c in history_set) or any(c in h or h in c for h in history_set)
            if not already_known:
                new_entities.append(c)

        # Check for abbreviations in this query
        abbrev_pairs, _ = self.resolver.resolve_query(query)

        # Case A: Pure follow-up within existing context (no new entities introduced OR referential pronoun used)
        if (len(new_entities) == 0 or has_referential_pronoun) and len(history_set) > 0:
            # Check if there are specific attribute triples for the history entity in the knowledge graph
            specific_triples = []
            for t in existing_list:
                t_text = f"{t.get('head', '')} {t.get('relation', '')} {t.get('tail', '')}".lower()
                if any(w in t_text for w in query_words if len(w) > 2 and w not in STOPWORDS):
                    specific_triples.append(t)

            active_facts = specific_triples if specific_triples else existing_list

            return {
                "query": query,
                "is_followup": True,
                "new_entities": [],
                "all_entities": list(history_set),
                "new_triples": [],
                "all_triples": existing_list,
                "active_triples": active_facts,
                "abbreviations": abbrev_pairs,
                "num_triples": len(active_facts)
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
