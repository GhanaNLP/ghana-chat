"""Head-focused Knowledge Graph Retrieval Engine for Ghana Chat.

Matches user noun phrases and entities strictly against the HEAD of knowledge graph triples
(and not the tail). Prioritizes specific multi-word entities over generic single words.
"""

import os
import re
from typing import List, Dict, Any, Optional
import pandas as pd

STOPWORDS = {
    "what", "who", "where", "when", "why", "how", "tell", "me", "about",
    "is", "are", "was", "were", "be", "been", "being", "the", "a", "an",
    "in", "on", "at", "of", "for", "to", "and", "or", "do", "does", "did",
    "can", "you", "give", "please", "with", "from", "by", "that", "this",
    "these", "those", "have", "has", "had"
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
        """Extract candidate noun phrases and named entities from query text."""
        candidates = []
        nlp = get_spacy_nlp()

        if nlp:
            doc = nlp(query)
            # 1. Noun chunks (stripped of leading/trailing stopwords)
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
        else:
            # Fallback simple heuristic tokenization
            clean_q = re.sub(r"[^\w\s-]", " ", query).lower()
            words = [w for w in clean_q.split() if w not in STOPWORDS and len(w) > 2]
            for i in range(len(words)):
                for j in range(i + 1, min(i + 4, len(words) + 1)):
                    candidates.append(" ".join(words[i:j]))

        # Sort longer (more specific) phrases first
        candidates.sort(key=lambda x: -len(x))
        return candidates

    def retrieve(self, query: str, max_triples: int = 10, country_filter: Optional[str] = "Ghana") -> Dict[str, Any]:
        """Retrieve triples where the candidate entity is strictly the HEAD."""
        candidates = self.extract_candidates(query)
        matched = []
        seen = set()

        # Primary pass: multi-word and exact phrase matches
        for cand in candidates:
            if cand in self.heads_map:
                for item in self.heads_map[cand]:
                    if country_filter and item["country"].lower() != country_filter.lower():
                        continue
                    t_key = (item["head"].lower(), item["relation"].lower(), item["tail"].lower())
                    if t_key not in seen:
                        seen.add(t_key)
                        matched.append(item)
                        if len(matched) >= max_triples:
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
