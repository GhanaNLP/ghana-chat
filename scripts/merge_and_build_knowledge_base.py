"""Merge GhanaQA vLLM extracted triples into the current Knowledge Graph.

Filters noise:
  - Drops conversational quotes & dialogue fragments (2nd person pronouns, speech verbs)
  - Drops ontology scaffolding (subclass of, instance of generic nouns)
  - Drops reflexive and quasi-reflexive pairs
  - Deduplicates entity pairs so each pair has 1 primary fact
  - Extracts dates from article URLs (YYYY-MM)
  - Merges with existing 58k facts and embeds with Model2Vec (potion-retrieval-32M)
"""
import os
import re
import json
import time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from model2vec import StaticModel

from ghana_chat.abbreviations import GHANA_ABBREVIATIONS

DATA_DIR = Path("/mnt/volume_d2wey28/projects/ghana-chat/data/verbalized_kg")
VLLM_JSONL = Path("/mnt/volume_d2wey28/projects/map-nav/kg_old_news_source/extracted_triples_vllm.jsonl")

SENT_PARQUET = DATA_DIR / "sentences.parquet"
EMB_NPY = DATA_DIR / "embeddings.npy"

CLEAN_DET = re.compile(r"^(the|a|an)\s+", re.I)
DATE_URL = re.compile(r"/(20\d\d)/(\d\d)/")
DATE_PAT = re.compile(r"\b(19\d\d|20\d\d)\b")

SPEECH_VERBS = {
    "said", "say", "says", "stated", "state", "states", "explain", "explained", "explains",
    "quoted", "quote", "told", "tell", "claimed", "claim", "warned", "warn", "urged", "urge",
    "called on", "reiterated", "noted", "revealed", "added", "clarified", "commented", "declared",
    "bemoaned", "admitted", "cautioned", "recalled", "remarked", "insisted"
}
QUOTE_TOKENS = {"you", "your", "we", "our", "i", "my", "me", "us"}
BAD_RELS = {
    "subclass of", "subclass_of", "instance of", "is a", "is an",
    "part of", "has part", "facet of", "is a facet of", "related to",
    "see also", "similar to", "type of", "category of", "has quality",
    "measures", "distinct from", "compared to", "opposite of", "different from"
}
GENERIC = {"he", "she", "it", "they", "we", "i", "someone", "people", "none", "all", "something", "anyone", "one"}

def main():
    t0 = time.time()
    print("Loading current verified facts from sentences.parquet...", flush=True)
    tab = pq.read_table(SENT_PARQUET)
    
    current_records = []
    seen_pairs = set()

    for r in tab.to_pylist():
        h = r["head"]
        t = r["tail"]
        hl = CLEAN_DET.sub("", h).strip().lower()
        tl = CLEAN_DET.sub("", t).strip().lower()
        pair_key = frozenset({hl, tl})
        seen_pairs.add(pair_key)
        current_records.append(r)

    print(f"Loaded {len(current_records):,} existing facts in {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    print(f"Reading and cleaning additional triples from {VLLM_JSONL}...", flush=True)
    new_records = []
    skipped = 0

    with open(VLLM_JSONL, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            r = json.loads(line)
            h = r.get("head", "").strip()
            rel = r.get("relation", "").strip()
            t = r.get("tail", "").strip()
            
            hl = CLEAN_DET.sub("", h).strip().lower()
            tl = CLEAN_DET.sub("", t).strip().lower()

            if len(hl) < 3 or len(tl) < 3 or hl == tl or hl in tl or tl in hl:
                skipped += 1
                continue
            if hl in GENERIC or tl in GENERIC:
                skipped += 1
                continue

            rel_norm = rel.lower().replace("_", " ").strip()
            if rel_norm in SPEECH_VERBS or rel_norm in BAD_RELS or len(rel_norm.split()) > 4 or len(rel_norm) == 0:
                skipped += 1
                continue

            if set(tl.split()) & QUOTE_TOKENS or len(h.split()) > 6 or len(t.split()) > 6:
                skipped += 1
                continue

            pair_key = frozenset({hl, tl})
            if pair_key in seen_pairs:
                skipped += 1
                continue

            seen_pairs.add(pair_key)

            # Extract date
            doc = r.get("doc", "") or ""
            m_url = DATE_URL.search(doc)
            dt = f"{m_url.group(1)}-{m_url.group(2)}" if m_url else ""
            if not dt:
                m_dt = DATE_PAT.findall(r.get("sentence", ""))
                if m_dt:
                    dt = m_dt[0]

            d_label = f"[Date: {dt}]" if dt else "[Date: 2025]"
            clean_rel = rel.replace("_", " ").strip()
            text_line = f'{d_label} ("{h}", "{clean_rel}", "{t}")'

            new_records.append({
                "head": h,
                "relation": clean_rel,
                "tail": t,
                "text": text_line,
                "date": dt,
                "doc_id": doc,
                "category": "news_triple"
            })

    print(f"Cleaned and accepted {len(new_records):,} new facts (skipped {skipped:,} noisy/duplicate items) in {time.time()-t0:.2f}s", flush=True)

    # Combine all records
    all_facts = current_records + new_records
    total = len(all_facts)
    print(f"Total merged facts: {total:,} ({len(current_records):,} existing + {len(new_records):,} new)", flush=True)

    # Assign IDs
    for idx, r in enumerate(all_facts):
        r["id"] = idx

    # Save to parquet
    out_tab = pa.table({
        "id": pa.array([r["id"] for r in all_facts], type=pa.int64()),
        "text": pa.array([r["text"] for r in all_facts], type=pa.string()),
        "head": pa.array([r["head"] for r in all_facts], type=pa.string()),
        "relation": pa.array([r["relation"] for r in all_facts], type=pa.string()),
        "tail": pa.array([r["tail"] for r in all_facts], type=pa.string()),
        "date": pa.array([r.get("date", "") for r in all_facts], type=pa.string()),
        "doc_id": pa.array([r.get("doc_id", "") for r in all_facts], type=pa.string()),
        "category": pa.array([r.get("category", "triple") for r in all_facts], type=pa.string())
    })
    pq.write_table(out_tab, SENT_PARQUET, compression="zstd")
    print(f"Saved {SENT_PARQUET} ({os.path.getsize(SENT_PARQUET)/(1024*1024):.1f} MB)", flush=True)

    # Embed with Model2Vec
    print(f"Embedding all {total:,} merged facts with potion-retrieval-32M...", flush=True)
    t0 = time.time()
    m2v = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    texts = [r["text"] for r in all_facts]
    embs = m2v.encode(texts, batch_size=4096)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = (embs / np.maximum(norms, 1e-9)).astype(np.float32)
    el = time.time() - t0
    print(f"Embedded in {el:.2f}s ({total/el:,.0f} facts/s)!", flush=True)

    np.save(EMB_NPY, embs)
    print(f"Saved {EMB_NPY} ({os.path.getsize(EMB_NPY)/(1024*1024):.1f} MB)", flush=True)
    print("Knowledge base merge and update complete!", flush=True)

if __name__ == "__main__":
    main()
