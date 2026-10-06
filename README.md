# Ghana Chat 🇬🇭

**An Open-Source, Customizable Grounded LLM Chat Assistant Built for Ghana's Context.**

[![GitHub License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/ghananlpcommunity/ghana-chat)
[![Model](https://img.shields.io/badge/Model-Gemma%204%202B%20%2F%20Qwen%202B-green.svg)](https://huggingface.co/google/gemma-4-E2B-it)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20H200-orange.svg)](https://www.nvidia.com/en-us/data-center/h200/)

Ghana Chat is an open-source, customizable conversational assistant built to provide truthful, factually grounded answers to questions about Ghana—spanning politics, governance, business, economic policies, institutions, public figures, history, and development.

Instead of relying on an LLM to hallucinate from memorized pretraining weights, Ghana Chat pairs an enriched **Ghanaian Knowledge Graph** with open-weight language models, enforcing strict factual grounding, chronological context, and clean abstention when records are unavailable.

Developers and organizations can clone this repository, add their own domain data (law, healthcare, agriculture, company records, or news), and deploy a customized assistant tailored to their specific needs.

---

## 🌟 Key Highlights

1. **Structured Knowledge Graph Grounding:**
   - Powered by an authoritative knowledge graph of structured facts (`("head", "relationship", "tail")`) covering Ghanaian institutions, people, policies, and events.
   - Cleaned to eliminate repetitive or quasi-reflexive statements, with deduplication ensuring every retrieved fact connects a distinct entity pair.

2. **Strict Exact Noun-Phrase & Alias Matching (No Fallback):**
   - User inputs are parsed into exact unified noun phrases and proper-noun titles (e.g. *"Coordinated Programme for Social and Economic Development"*, *"Bank of Ghana"*, *"Telecel Ghana"*).
   - Strict matching ensures that if an entity is not in the knowledge graph, the system **abstains cleanly** instead of using fuzzy semantic fallbacks to pull unrelated facts.

3. **Bidirectional Abbreviation Resolution:**
   - Over 100 Ghanaian public bodies, ministries, and political parties (`NDPC`, `EC`, `CPP`, `NPP`, `NDC`, `GRA`, `SSNIT`, `COCOBOD`, `KNUST`, etc.) are mapped bidirectionally.
   - Querying *"NDC"* automatically matches *"National Democratic Congress"* and vice versa.
   - The authoritative definition (e.g. `[Definition] ("NDC", "abbreviation of", "National Democratic Congress")`) is automatically injected as **Source Item #1** in the LLM's context.

4. **Temporal Awareness (`[Date: YYYY-MM-DD]` & Today's Date):**
   - Each fact in the context is labeled with the publication date of its source article.
   - Today's date is dynamically provided to the model in the system prompt.
   - Allows the LLM to establish accurate **relative temporal context** (e.g. distinguishing whether an exchange rate, policy statement, or appointment occurred recently, last year, or historically).

5. **Real-Time Token Streaming (Server-Sent Events):**
   - Supports low-latency token-by-token streaming via SSE.
   - First token renders in **~1.8 seconds**, while the collapsible **"View Knowledge Facts"** drawer displays the underlying facts upon completion.

6. **Easily Customizable & Model-Agnostic:**
   - Swap any Hugging Face open-weight model with a single environment variable (`GHANA_CHAT_MODEL`).
   - Add your own custom articles or datasets using the automated enrichment pipeline in `scripts/enrich_knowledge_base.py`.

---

## ⚠️ Model Parameter Recommendation (Avoiding Hallucinations)

Ghana Chat is designed to be model-agnostic, allowing you to run any compatible model. However, based on our empirical testing on the NVIDIA H200 GPU, **we strongly recommend selecting models with at least 2B+ parameters**:

| Parameter Tier | Tested Models | Grounded QA Behavior on Ghana Context |
|---|---|---|
| **Sub-1B (~350M–800M)** | `IBM Granite 4.0 350M`, `Qwen 0.5B`, `SmolLM 135M` | ⚠️ **Hallucination Risk:** Sub-1B models frequently struggle with cross-sentence entity attribution. In our benchmark on the CPP founder question, a 350M model confused a former general secretary in passage [1] with the party founder in passage [4], incorrectly attributing the founding to the general secretary. Tends to copy-paste long verbatim chunks without synthesis. |
| **2B+ Parameters (Recommended)** | `Google Gemma 4 2B` ⭐, `Qwen 3.5 2B`, `Qwopus 3.5 2B` | ✅ **Reliable Grounding:** Accurately cross-references multi-hop facts, correctly attributes roles across distinct items, synthesizes balanced 60–120 word narrative paragraphs, and cleanly abstains when facts are absent. |

**Default Recommended Model:** `google/gemma-4-E2B-it` (or `Qwen/Qwen3.5-2B`).

---

## 🛠️ Quickstart

### 1. Installation

```bash
git clone https://github.com/GhanaNLP/ghana-chat.git
cd ghana-chat
pip install -e .
python -m spacy download en_core_web_sm
```

### 2. Run Locally or on Remote GPU

```bash
# Set your desired model (defaults to google/gemma-4-E2B-it)
export GHANA_CHAT_MODEL="google/gemma-4-E2B-it"

# Start the API and Web Interface
python -m uvicorn ghana_chat.server:app --host 0.0.0.0 --port 8000
```

- **Interactive Web Interface:** Open `http://localhost:8000` in your browser.
- **Interactive API Documentation:** Open `http://localhost:8000/docs`.

---

## 🔌 API Usage

### `POST /ask` (Standard or Streaming)

#### Streaming Request (Server-Sent Events)
```bash
curl -N -s -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Who is the chairperson of the NDPC?",
    "stream": true
  }'
```

The response streams events in real time:
- `event: meta`: Returns the matched knowledge facts and grounding status.
- `event: token`: Returns generated tokens incrementally.
- `event: done`: Emits completion status.

#### Standard JSON Request
```bash
curl -s -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"message": "Who is the chairperson of the NDPC?"}'
```

**Response:**
```json
{
  "question": "Who is the chairperson of the NDPC?",
  "answer": "Dr. Nii Moi Thompson is the Chairman of the National Development Planning Commission (NDPC). The commission is a government agency located in Accra, Ghana, and operates as part of the Government of Ghana.",
  "grounded": true,
  "sources": [
    {
      "head": "NDPC",
      "relation": "abbreviation of",
      "tail": "National Development Planning Commission",
      "date": "Definition",
      "score": 100.0
    },
    {
      "head": "National Development Planning Commission (NDPC)",
      "relation": "country",
      "tail": "Ghana",
      "date": "2025-09-05",
      "score": 10.0
    }
  ],
  "model": "google/gemma-4-E2B-it"
}
```

---

## 📚 Enriching the Knowledge Base with Your Own Data

You can easily enrich or customize the knowledge base with your own articles, documents, or pre-extracted triples using `scripts/enrich_knowledge_base.py`.

### 1. From Raw Articles (CSV or JSONL)
Provide a CSV or JSONL file containing your articles (columns: `content`, `date`, `url`, `title`):

```bash
# Extract triples using Mistral API:
python scripts/enrich_knowledge_base.py \
  --input data/my_articles.csv \
  --api-provider mistral \
  --api-key $MISTRAL_API_KEY

# Extract triples using a local vLLM or Ollama instance:
python scripts/enrich_knowledge_base.py \
  --input data/my_articles.jsonl \
  --api-provider vllm \
  --api-url "http://localhost:8000/v1/chat/completions" \
  --model "Qwen/Qwen2.5-7B-Instruct"
```

### 2. Ingesting Pre-Extracted Triples
If you already have triples formatted as `("head", "relation", "tail")`:

```bash
python scripts/enrich_knowledge_base.py \
  --triples-file "data/my_triples.txt" \
  --meta-file "data/row_dates.json"
```

The pipeline automatically:
- Filters reflexive and quasi-reflexive statements.
- Deduplicates entity pairs so each pair has only one distinct relationship.
- Maps article publication dates (`[Date: YYYY-MM-DD]`).
- Injects abbreviation assertions for Ghanaian public bodies.
- Builds the memory-mapped parquet and Model2Vec embeddings (`potion-retrieval-32M`).

See the complete guide in [`docs/ENRICHING_KNOWLEDGE.md`](docs/ENRICHING_KNOWLEDGE.md).

---

## 🚀 Deployment

### Live Hugging Face Space
The web client is published as a static Hugging Face Space under `space/`. To deploy updates:

```bash
python scripts/publish_space.py --repo-id ghananlpcommunity/ghana-chat
```

View the live Space at: **[huggingface.co/spaces/ghananlpcommunity/ghana-chat](https://huggingface.co/spaces/ghananlpcommunity/ghana-chat)**

### NVIDIA H200 GPU Supervisor
For persistent 24/7 deployment on remote GPU instances (e.g. H200), run the automated supervisor:

```bash
bash scripts/run_api_supervisor.sh
```

The supervisor automatically recovers the Uvicorn worker in case of transient GPU memory faults or unexpected crashes.

---

## 📄 License

Apache-2.0
