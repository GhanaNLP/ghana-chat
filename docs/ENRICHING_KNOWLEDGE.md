# Enriching and Customizing the Knowledge Base

Ghana Chat is designed as an **open-source, customizable grounded LLM assistant** built for the Ghanaian context. Rather than relying on a fixed, static dataset, the knowledge base can be continuously updated, expanded, and customized with your own domain data—such as parliamentary proceedings, legal statutes, company reports, academic papers, local government news, or health policies.

---

## 1. How Knowledge Triples Are Extracted

The knowledge base consists of structured **(head, relationship, tail)** assertions. Triples in the core dataset were extracted using high-capacity LLMs prompted with a targeted relationship taxonomy:

```
Extract a structured knowledge graph from the following text. Format the output as a list of triples.
Focus on these relationships:
  instance of, subclass of, part of, has part, located in, country, owned by, creator,
  author, participant in, employer, member of, part of a series, subsidiary, inception, end time.

Format the output exactly as:
    ("entity", "relationship", "entity/value"),
    ("entity", "relationship", "entity/value"),

Only include the triple lines, no additional text.
```

### Supported Relationship Taxonomy

| Relationship | Meaning / Usage | Example |
|---|---|---|
| `instance of` | Class / category membership | `("Bank of Ghana", "instance of", "central bank")` |
| `country` | Sovereign national location | `("cedi", "country", "Ghana")` |
| `located in` | City, region, or territorial entity | `("NDPC", "located in", "Accra")` |
| `employer` | Employment or executive appointment | `("Bank of Ghana", "employer", "Dr. Johnson Asiama")` |
| `member of` | Institutional or political membership | `("Johnson Asiedu Nketiah", "member of", "NDC")` |
| `part of` | Component of a broader system or sector | `("Ghana Cedi", "part of", "monetary system of Ghana")` |
| `creator` | Legislative, regulatory, or policy creator | `("Bank of Ghana", "creator", "measures to stabilize cedi")` |
| `subsidiary` | Corporate or organizational subsidiary | `("Telecel Ghana", "subsidiary", "Telecel Group")` |
| `inception` | Establishment or founding date | `("Agricultural Development Bank", "inception", "1965")` |
| `abbreviation of` | Official acronym expansion | `("NDPC", "abbreviation of", "National Development Planning Commission")` |

---

## 2. Ingesting Your Own Data

You can enrich the knowledge base in two ways:
1. **From raw articles (CSV or JSONL)**: The automated pipeline sends articles to an LLM extraction API and ingests the resulting triples.
2. **From pre-extracted triples (TXT)**: If you already extracted triples, you can ingest them directly.

### Input Data Format (CSV / JSONL)

Prepare a CSV or JSONL file with the following columns:

| Column | Required | Description | Example |
|---|:---:|---|---|
| `content` (or `text`) | **Yes** | Full article or document body | *"Telecel Ghana has announced its 2025 SME Month initiative..."* |
| `date` | Optional | Publication date (`YYYY-MM-DD` or `Month YYYY`) | `2025-09-05` |
| `url` | Optional | Source link or document identifier | `https://citinewsroom.com/2025/09/article` |
| `title` | Optional | Article headline | *"Telecel launches SME Month in Accra"* |

---

## 3. Running the Enrichment Pipeline

We provide a single command-line pipeline in `scripts/enrich_knowledge_base.py`.

### Option A: Extract Triples from New Articles via LLM API

You can use **Mistral**, **OpenAI**, a local **vLLM** instance, or **Ollama**:

```bash
# Using Mistral API:
python scripts/enrich_knowledge_base.py \
  --input data/new_articles.csv \
  --api-provider mistral \
  --api-key $MISTRAL_API_KEY

# Using a local vLLM or Ollama instance:
python scripts/enrich_knowledge_base.py \
  --input data/new_articles.jsonl \
  --api-provider vllm \
  --api-url "http://localhost:8000/v1/chat/completions" \
  --model "Qwen/Qwen2.5-7B-Instruct"
```

### Option B: Ingesting Pre-Extracted Triples

If you have a text file of triples formatted as `("head", "relation", "tail")` (like `data/knowledge_graph_output.txt`):

```bash
python scripts/enrich_knowledge_base.py \
  --triples-file "data/knowledge_graph_output.txt" \
  --meta-file "data/row_dates.json"
```

---

## 4. Automatic Processing Applied by the Pipeline

When the pipeline runs, it automatically executes the following data hygiene steps:

1. **Reflexive & Quasi-Reflexive Filtering:**
   - Filters out triples where the head equals the tail (`head == tail`).
   - Filters out quasi-reflexive statements where one entity is a sub-phrase of the other (e.g. dropping `"Telecel Ghana CEO employs Telecel Ghana"` or `"NDPC guidelines created NDPC guidelines"`).
2. **Entity-Pair Deduplication:**
   - For every distinct pair `{head, tail}`, only the single most informative, specific relationship is retained.
   - Eliminates redundant statements (e.g. removes repeating `(cedi, Ghana)` 10 times).
3. **Publication Date Attachment:**
   - Stamped with the source article's publication date:
     `[Date: 2025-09-05] ("cedi", "part of a series", "cedi exchange rate in 2025")`
   - Gives the LLM explicit chronological context.
4. **Abbreviation Assertion Injection:**
   - Injects clean definition assertions for over 100 Ghanaian public bodies:
     `[Definition] ("NDPC", "abbreviation of", "National Development Planning Commission")`
5. **Model2Vec Index Refresh:**
   - Re-embeds all clean facts with `minishlab/potion-retrieval-32M` (takes ~4 seconds for 60,000 facts) and saves `data/verbalized_kg/sentences.parquet` and `data/verbalized_kg/embeddings.npy`.

Once complete, restart the API server (`ghana-chat serve`), and the updated knowledge is immediately live.

---

## 5. Model Parameter Recommendations to Avoid Hallucinations

Ghana Chat is model-agnostic. You can switch models by setting the environment variable `GHANA_CHAT_MODEL`:

```bash
export GHANA_CHAT_MODEL="google/gemma-4-E2B-it"
# or
export GHANA_CHAT_MODEL="Qwen/Qwen3.5-2B"
# or
export GHANA_CHAT_MODEL="Jackrong/Qwopus3.5-2B-v3"
```

### Empirical Recommendation: Use Models with at least 2B+ Parameters

Based on our empirical benchmarks on the NVIDIA H200 GPU, **we strongly recommend deploying models with at least 2B parameters** for grounded question answering:

| Parameter Tier | Tested Models | Typical Grounded QA Behavior |
|---|---|---|
| **Sub-1B (~350M–800M)** | `IBM Granite 4.0 350M`, `Qwen 0.5B`, `SmolLM 135M` | ⚠️ **Hallucination Risk:** Struggles with cross-sentence entity attribution. In our tests on the CPP founder question, a 350M model confused a former general secretary in passage [1] with the party founder in passage [4], incorrectly attributing the founding to the secretary. Tends to copy-paste long verbatim chunks. |
| **2B+ Parameters** | `Google Gemma 4 2B`, `Qwen 3.5 2B`, `Qwopus 3.5 2B` | ✅ **Reliable Grounding:** Accurately cross-references multi-hop facts, correctly resolves roles, synthesizes clean 60–120 word narrative paragraphs, and cleanly abstains when facts are absent. |

Models in the 2B tier run at high generation speeds (~15–20 tokens/sec on modern GPUs) while using only ~4.5 GB of VRAM in `bfloat16`, delivering the optimal balance of speed, memory efficiency, and strict factual compliance.
