# Ghana Chat 🇬🇭

**Head-Focused Knowledge Graph Grounded QA powered by Qwen 2B on NVIDIA H200**

Ghana Chat is a truthful question-answering assistant for Ghana. Instead of training or fine-tuning models to memorize facts, Ghana Chat pairs a structured **Ghana Knowledge Graph** with **Qwen 2B** (`Qwen/Qwen3.5-2B`), enforcing strict factual grounding.

---

## Key Features

1. **Head-Focused Knowledge Graph Retrieval:**
   - User queries are parsed into noun phrases and named entities.
   - Matching is performed **strictly against the `head`** of the knowledge graph (and not the tail), ensuring the retrieved facts describe the exact entity being queried.
   - Self-loops, reflexive edges, and pronouns are filtered out.
2. **Zero-Hallucination Grounding:**
   - The generator is strictly instructed to answer *only* from the provided graph facts.
   - If an entity or fact is not in the knowledge graph, the model explicitly admits it does not have the information rather than guessing.
3. **Optimized for NVIDIA H200:**
   - Generates at **~28.8 tokens/sec** on an H200 GPU in native `bfloat16`.
   - Outperforms smaller sub-1B models both in factual compliance and inference throughput.
4. **Deployable API & Static Web Client:**
   - Single unified FastAPI server with CORS enabled.
   - Static HTML web interface ready for self-hosting or deployment to Hugging Face Spaces.

---

## Benchmark: Small LLM Comparison on H200

We benchmarked candidate small LLMs on Knowledge Graph-grounded generation using the NVIDIA H200 GPU:

| Model | Parameters | Factual Pass Rate | Speed on H200 | Behavior & Grounding Compliance |
|---|:---:|:---:|:---:|---|
| **Qwen/Qwen3.5-2B** ⭐ | **2B** | **100.0%** | **28.8 tok/s** | **Top Performer:** Flawless fact retrieval, zero hallucinations, perfect refusal on unknown queries, clean formatting. |
| **Qwen/Qwen3.5-0.8B** | 0.8B | 100.0% | 13.6 tok/s | 100% accurate, but slower due to linear convolution overhead on Hopper. |
| **google/gemma-3-1b-it** | 1B | 80.0% | 21.9 tok/s | Accurate, but suffered from repetition loops at sequence ends. |
| **SmolLM2-135M-Instruct** | 135M | 80.0% | 12.3 tok/s | Good fact extraction, but failed negative constraints (hallucinated on missing facts). |
| **Qwen2.5-0.5B-Instruct** | 0.5B | 60.0% | 16.9 tok/s | Hallucinated outside knowledge when triples did not contain the answer. |
| **Falcon-H1-Tiny-100M** | 100M | 20.0% | 11.3 tok/s | Severe hallucinations (invented fake countries/agencies); lacks capacity for strict grounding. |

> **Why Qwen 2B is faster than 0.8B on H200:**  
> On Hopper architectures (H200), single-sequence generation is memory-latency bound. A 2048 hidden dimension saturates Hopper's Tensor Cores and 4.8 TB/s HBM3e bandwidth, whereas sub-1B models suffer from kernel dispatch overhead.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/GhanaNLP/ghana-chat.git
cd ghana-chat
pip install -e .
python -m spacy download en_core_web_sm
```

### 2. Command Line (CLI)

Ask any question directly from your terminal:

```bash
ghana-chat ask "Who is the chairperson of the NDPC, and where is its headquarters located?"
```

### 3. API & Web Interface Server

Start the server locally or on your remote GPU instance:

```bash
ghana-chat serve --host 0.0.0.0 --port 8000
```

- **Interactive Web Interface:** Open `http://localhost:8000` in your browser.
- **Interactive API Documentation:** Open `http://localhost:8000/docs`.

---

## API Endpoints

### `POST /ask`
Submit a question and receive a grounded answer with the underlying Knowledge Graph triples.

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is the chairperson of the NDPC?"}'
```

**Response:**
```json
{
  "question": "Who is the chairperson of the NDPC?",
  "answer": "Based on the provided Knowledge Graph facts:\n\n* The chairperson of the NDPC is George Gyan-Baffour.\n* The headquarters location of the NDPC is Accra.",
  "triples": [
    {"head": "NDPC", "relation": "chairperson", "tail": "George Gyan-Baffour", "country": "Ghana"},
    {"head": "NDPC", "relation": "headquarters location", "tail": "Accra", "country": "Ghana"},
    {"head": "NDPC", "relation": "country", "tail": "Ghana", "country": "Ghana"}
  ],
  "entities": ["ndpc", "chairperson"],
  "latency_s": 0.42,
  "tokens_generated": 43,
  "tokens_per_sec": 28.8,
  "model": "Qwen/Qwen3.5-2B"
}
```

### `POST /retrieve`
Retrieve head-matched Knowledge Graph triples without running LLM generation:

```bash
curl -X POST "http://localhost:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"question": "National Development Planning Commission"}'
```

### `GET /health`
Verify server status, GPU hardware, and indexed head entities:

```bash
curl http://localhost:8000/health
```

---

## Deployment on NVIDIA H200

For best inference performance (~28 tok/s), deploy on your **Ghana NLP H200 GPU**:

```bash
# Connect to H200 instance
ssh h200

# Clone and setup
cd /mnt/volume_d2wey28/projects
git clone https://github.com/GhanaNLP/ghana-chat.git
cd ghana-chat
source .venv/bin/activate
pip install -e .

# Run with runner script
bash deploy/run_h200.sh 8000 0.0.0.0
```

To run as a systemd service:
```bash
sudo cp deploy/ghana-chat.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ghana-chat
```

---

## Static Hugging Face Space

The web UI is packaged as a standalone static Space under `space/`. To publish:

```bash
python scripts/publish_space.py --repo-id ghananlpcommunity/ghana-chat
```

Users can use the static Space in their browser and point it to their deployed H200 API endpoint.

---

## License

Apache-2.0
