"""Enrich the Ghana Chat Knowledge Base with new articles, documents, or triples.

This script implements the complete extraction and enrichment pipeline:
1. Ingests raw articles from CSV or JSONL (or pre-extracted triples).
2. Extracts structured (head, relation, tail) triples via LLM API (Mistral, OpenAI, vLLM, or Ollama).
3. Attaches publication dates and URLs from article metadata.
4. Cleans and deduplicates entity pairs, eliminating reflexive and redundant relationships.
5. Injects authoritative Ghanaian abbreviation definitions.
6. Builds/updates the binary parquet and Model2Vec embeddings for instant, sub-millisecond retrieval.

Usage:
  # Extract from raw CSV articles using Mistral API:
  python scripts/enrich_knowledge_base.py --input data/new_articles.csv --api-provider mistral --api-key $MISTRAL_API_KEY

  # Ingest pre-extracted triples and rebuild the knowledge base:
  python scripts/enrich_knowledge_base.py --triples-file "data/knowledge_graph_output.txt" --meta-file "data/row_dates.json"
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Default output locations
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = BASE_DIR / "data" / "verbalized_kg"
DEFAULT_KG_TXT = BASE_DIR / "data" / "knowledge_graph_output.txt"

TRIPLE_PAT = re.compile(r"^\s*\(\s*\"([^\"]+)\"\s*,\s*\"([^\"]+)\"\s*,\s*\"([^\"]+)\"\s*\)\s*,?\s*$")
ROW_PAT = re.compile(r"//\s*Result for row\s*(\d+):")
CLEAN_DET = re.compile(r"^(the|a|an)\s+", re.I)

EXTRACTION_SYSTEM_PROMPT = """Extract a structured knowledge graph from the following text. Format the output as a list of triples. Focus on these relationships: instance of, subclass of, part of, has part, located in, country, owned by, creator, author, participant in, employer, member of, part of a series, subsidiary, inception, end time.

Format the output exactly as:
    ("entity", "relationship", "entity/value"),
    ("entity", "relationship", "entity/value"),

Only include the triple lines, no additional text."""


def extract_triples_via_llm(
    text: str,
    provider: str = "mistral",
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model: str = "mistral-large-latest"
) -> List[Tuple[str, str, str]]:
    """Call an LLM API to extract knowledge graph triples from raw text."""
    import requests

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if provider == "mistral":
        url = api_url or "https://api.mistral.ai/v1/chat/completions"
    elif provider in ("openai", "vllm", "ollama"):
        url = api_url or "https://api.openai.com/v1/chat/completions"
    else:
        url = api_url

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        "temperature": 0.1
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=45)
        res.raise_for_status()
        raw_content = res.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"Extraction API call failed: {exc}", file=sys.stderr)
        return []

    triples = []
    for line in raw_content.splitlines():
        m = TRIPLE_PAT.match(line.strip())
        if m:
            h, r, t = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            triples.append((h, r, t))
    return triples


def verbalize(h: str, r: str, t: str) -> str:
    """Verbalize a triple into a natural English statement."""
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


def process_and_deduplicate(
    raw_triples: List[Tuple[str, str, str, str, str]]
) -> List[Dict[str, Any]]:
    """Clean, filter quasi-reflexive triples, and deduplicate entity pairs."""
    pair_seen = {}
    for h, r, t, dt, url in raw_triples:
        hl = CLEAN_DET.sub("", h).strip().lower()
        tl = CLEAN_DET.sub("", t).strip().lower()

        # Filter reflexive and quasi-reflexive triples
        if not hl or not tl or hl == tl or hl in tl or tl in hl:
            continue

        pair_key = frozenset({hl, tl})
        if pair_key not in pair_seen:
            pair_seen[pair_key] = (h, r, t, dt, url)
        else:
            prev_item = pair_seen[pair_key]
            prev_dt = prev_item[3]
            # Prefer newer dates or specific relations
            if dt and (not prev_dt or dt > prev_dt):
                pair_seen[pair_key] = (h, r, t, dt, url)

    records = []
    for h, r, t, dt, url in pair_seen.values():
        records.append({
            "head": h,
            "relation": r,
            "tail": t,
            "date": dt,
            "doc_id": url,
            "category": "triple"
        })
    return records


def build_knowledge_artifacts(
    records: List[Dict[str, Any]],
    output_dir: Path
):
    """Build verbalized parquet and Model2Vec embeddings."""
    from ghana_chat.abbreviations import GHANA_ABBREVIATIONS
    try:
        from model2vec import StaticModel
    except ImportError:
        print("Please install model2vec: pip install model2vec", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    sent_parquet = output_dir / "sentences.parquet"
    emb_npy = output_dir / "embeddings.npy"

    # Inject abbreviation facts
    abbrev_facts = []
    for abbr, full in GHANA_ABBREVIATIONS.items():
        abbrev_facts.append({
            "head": abbr,
            "relation": "abbreviation of",
            "tail": full,
            "date": "Definition",
            "doc_id": "abbrev_registry",
            "category": "abbreviation"
        })

    all_records = records + abbrev_facts
    total = len(all_records)
    print(f"Total knowledge facts: {total:,} ({len(records):,} triples + {len(abbrev_facts)} abbreviation assertions)")

    texts = []
    for r in all_records:
        d_label = f"[Date: {r['date']}]" if r["date"] and r["date"] != "Definition" else "[Definition]"
        texts.append(f'{d_label} ("{r["head"]}", "{r["relation"]}", "{r["tail"]}")')

    tab = pa.table({
        "id": pa.array(list(range(total)), type=pa.int64()),
        "text": pa.array(texts, type=pa.string()),
        "head": pa.array([r["head"] for r in all_records], type=pa.string()),
        "relation": pa.array([r["relation"] for r in all_records], type=pa.string()),
        "tail": pa.array([r["tail"] for r in all_records], type=pa.string()),
        "date": pa.array([r["date"] for r in all_records], type=pa.string()),
        "doc_id": pa.array([r["doc_id"] for r in all_records], type=pa.string()),
        "category": pa.array([r["category"] for r in all_records], type=pa.string())
    })
    pq.write_table(tab, sent_parquet, compression="zstd")
    print(f"Saved {sent_parquet} ({os.path.getsize(sent_parquet)/(1024*1024):.1f} MB)")

    print("Embedding knowledge facts with potion-retrieval-32M...")
    t0 = time.time()
    m2v = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    embs = m2v.encode(texts, batch_size=2048)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = (embs / np.maximum(norms, 1e-9)).astype(np.float32)
    el = time.time() - t0
    print(f"Embedded in {el:.2f}s ({total/el:,.0f} facts/s)!")

    np.save(emb_npy, embs)
    print(f"Saved {emb_npy} ({os.path.getsize(emb_npy)/(1024*1024):.1f} MB)")
    print("Knowledge base updated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Enrich Ghana Chat Knowledge Base with custom data")
    parser.add_argument("--input", help="Path to input CSV or JSONL file with articles (columns: content, date, url, title)")
    parser.add_argument("--triples-file", default=str(DEFAULT_KG_TXT), help="Path to existing knowledge_graph_output.txt file")
    parser.add_argument("--meta-file", help="Optional path to row_dates.json for mapping article dates to row indices")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for verbalized_kg artifacts")
    parser.add_argument("--api-provider", default="mistral", choices=["mistral", "openai", "vllm", "ollama"], help="LLM provider for extraction")
    parser.add_argument("--api-key", default=os.environ.get("MISTRAL_API_KEY") or os.environ.get("OPENAI_API_KEY"), help="API key for extraction")
    parser.add_argument("--api-url", help="Custom API endpoint URL for vLLM or Ollama")
    parser.add_argument("--model", default="mistral-large-latest", help="LLM model name for extraction")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    triples_to_process = []

    # 1. Ingest existing triples file if available
    triples_path = Path(args.triples_file)
    row_meta = {}
    if args.meta_file and Path(args.meta_file).exists():
        with open(args.meta_file, "r") as f:
            row_meta = json.load(f)

    if triples_path.exists():
        print(f"Ingesting triples from {triples_path}...")
        current_date = ""
        current_url = ""
        with open(triples_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                l = line.strip()
                m_row = ROW_PAT.match(l)
                if m_row:
                    row_idx = m_row.group(1)
                    meta = row_meta.get(row_idx, {})
                    current_date = meta.get("date", "")
                    current_url = meta.get("url", f"row_{row_idx}")
                    continue
                m = TRIPLE_PAT.match(l)
                if m:
                    h, r, t = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                    triples_to_process.append((h, r, t, current_date, current_url))

    # 2. Extract new triples from input articles if provided
    if args.input:
        in_path = Path(args.input)
        print(f"Reading new articles from {in_path}...")
        import pandas as pd
        if in_path.suffix.lower() == ".csv":
            df = pd.read_csv(in_path)
        else:
            df = pd.read_json(in_path, lines=True)

        content_col = "content" if "content" in df.columns else ("text" if "text" in df.columns else None)
        if not content_col:
            raise ValueError(f"Input file must contain a 'content' or 'text' column. Found: {df.columns.tolist()}")

        for idx, row in df.iterrows():
            text = str(row[content_col]).strip()
            if len(text) < 30:
                continue
            date_val = str(row.get("date", ""))
            url_val = str(row.get("url", f"custom_doc_{idx}"))
            print(f"Extracting triples from article {idx + 1}/{len(df)}...")
            extracted = extract_triples_via_llm(
                text=text,
                provider=args.api_provider,
                api_key=args.api_key,
                api_url=args.api_url,
                model=args.model
            )
            for h, r, t in extracted:
                triples_to_process.append((h, r, t, date_val, url_val))
            time.sleep(0.5)

    if not triples_to_process:
        print("No triples available to process. Provide --input or --triples-file.", file=sys.stderr)
        return

    # 3. Clean, deduplicate entity pairs, and update artifacts
    records = process_and_deduplicate(triples_to_process)
    build_knowledge_artifacts(records, out_dir)


if __name__ == "__main__":
    main()
