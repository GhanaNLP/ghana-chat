"""Configuration settings for Ghana Chat."""

import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

MODEL_ID = os.environ.get("GHANA_CHAT_MODEL", "google/gemma-4-E2B-it")
KG_PATH = os.environ.get("GHANA_CHAT_KG_PATH", str(DATA_DIR / "ghana_triples.parquet"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

DEFAULT_HOST = os.environ.get("GHANA_CHAT_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("GHANA_CHAT_PORT", "8000"))

MAX_RETRIEVED_TRIPLES = int(os.environ.get("MAX_RETRIEVED_TRIPLES", "10"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))

# ---------------------------------------------------------------------------
# Knowledge graph hygiene
# ---------------------------------------------------------------------------
# Relations produced by LLM extraction that encode lexical/ontology scaffolding
# rather than real-world facts. These are filtered out at retrieval time so the
# generator never narrates them (e.g. "farmers is a subclass of women").
ONTOLOGY_NOISE_RELATIONS = {
    "subclass of", "subclass_of", "is a subclass of", "instance of",
    "is_a", "isa", "is an instance of", "has part", "part of",
    "field of this occupation", "field of work", "field of the occupation",
    "opposite of", "different from", "has part(s)", "is a facet of",
    "facet of", "related to", "see also", "similar to", "type of",
    "category of", "is a category of", "has quality", "quality of",
    "is a measure of", "measures", "distinct from", "compared to",
}


def _normalize_rel_key(rel: str) -> str:
    """Normalise a relation to lowercase space-separated form for comparisons."""
    import re as _re
    out = str(rel).strip().lower().replace("-", " ").replace("_", " ")
    return _re.sub(r"\s+", " ", out).strip()


# Pre-normalised lookup so callers can compare against normalised relations.
ONTOLOGY_NOISE_KEYS = frozenset(
    _normalize_rel_key(r) for r in ONTOLOGY_NOISE_RELATIONS
)


# Candidate words that describe the *question* rather than an entity in the graph.
QUESTION_TERM_STOPLIST = {
    "month", "year", "day", "week", "time", "date", "many", "much", "few",
    "people", "person", "man", "woman", "way", "thing", "things", "lot",
    "number", "amount", "part", "kind", "sort", "type", "name", "place",
    "area", "region", "country", "city", "town", "village", "home",
    "government", "state", "group", "level", "side", "end", "top", "bottom",
}
