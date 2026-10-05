"""Knowledge Graph Grounded Response Generator using Qwen 2B."""

import re
import time
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def normalize_relation_key(relation: str) -> str:
    """Lowercase/underscore-normalised relation used for ontology-noise filtering."""
    rel = str(relation).strip().lower()
    rel = rel.replace("-", " ").replace("_", " ")
    rel = re.sub(r"\s+", " ", rel).strip()
    return rel


def _clean_relation(relation: str) -> str:
    """Turn a schema-style relation into a readable verb phrase.

    Casing is preserved so acronyms inside relations (CEO, NDC, GNPC) survive.
    """
    rel = str(relation).strip()
    rel = rel.replace("_", " ")
    rel = re.sub(r"\s+", " ", rel).strip()
    # Collapse duplicated copulas produced by schemas like "is_is_a".
    rel = re.sub(r"\b(is|are|was|were)\s+(is|are|was|were)\b", r"\1", rel, flags=re.I)
    return rel.strip()



def verbalize_triple(head: str, relation: str, tail: str) -> str:
    """Render a triple as a single natural sentence.

    Following the GraphQA approach, the model is shown plain English statements
    (e.g. "Apple announced the Vision Pro in 2023.") instead of schema rows
    (e.g. "Apple | announced | the Vision Pro"). This stops the generator from
    echoing raw ontology scaffolding into its prose.
    """
    head = str(head).strip()
    tail = str(tail).strip()
    rel = _clean_relation(relation)
    if not rel:
        return f"{head} {tail}."
    sentence = f"{head} {rel} {tail}".strip()
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(r"\s+([,.;:!?])", r"\1", sentence)
    # Capitalise only when the first word is entirely lowercase (never mangle acronyms like "iOS")
    first = sentence.split(" ", 1)[0]
    if first.islower() and first.isalpha():
        sentence = sentence[0].upper() + sentence[1:]
    if not sentence.endswith((".", "!", "?", "…")):
        sentence += "."
    return sentence


class GroundedGenerator:
    """Loads Gemma 4 2B and generates answers strictly grounded on retrieved source passages."""

    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        from . import config
        self.model_id = model_id or config.MODEL_ID
        self.device = device or config.DEVICE
        self.dtype = config.DTYPE

        print(f"Loading {self.model_id} on {self.device} ({self.dtype})...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True
        ).eval()
        print(f"Loaded {self.model_id} in {time.time() - t0:.2f}s")

    def format_triples(self, triples: List[Dict[str, Any]]) -> str:
        """Render triples as natural sentences for the prompt."""
        from . import config
        if not triples:
            return "No knowledge graph facts found for this query."

        lines = []
        seen = set()
        for t in triples:
            if normalize_relation_key(t.get("relation", "")) in config.ONTOLOGY_NOISE_KEYS:
                continue
            sentence = verbalize_triple(t.get("head", ""), t.get("relation", ""), t.get("tail", ""))
            if sentence in seen:
                continue
            seen.add(sentence)
            lines.append(f"- {sentence}")

        if not lines:
            return "No usable knowledge graph facts found for this query."
        return "\n".join(lines)

    def extract_entities(self, query: str, max_entities: int = 6) -> List[str]:
        """LLM entity extraction, mirroring the GraphQA "Entities Extracted" step.

        Returns a short list of entity strings, or an empty list if the model
        fails to produce anything parseable (the caller then falls back to the
        deterministic spaCy extractor).
        """
        from . import config

        prompt = (
            "<|im_start|>system\n"
            "You extract entity names from questions. Reply with a comma-separated "
            "list of the proper nouns, organisations, places and key concepts the "
            "question is about. Include only names that appear in the question. "
            "Never include question words, pronouns or generic nouns. "
            "If there are none, reply exactly: NONE<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        text = text.split("</think>")[-1].strip()

        if not text or text.upper().startswith("NONE"):
            return []

        ents = []
        for part in re.split(r"[,\n;]+", text):
            e = part.strip().strip('."\'')
            e = re.sub(r"^\d+[.)]\s*", "", e).strip()
            if not e or len(e) < 2 or len(e) > 80:
                continue
            if e.lower() in config.QUESTION_TERM_STOPLIST:
                continue
            # Only keep entities actually present in the question (guards hallucination)
            if e.lower() not in query.lower():
                continue
            if e not in ents:
                ents.append(e)
            if len(ents) >= max_entities:
                break
        return ents

    def format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Render retrieved sentences as a numbered, citable context block."""
        if not sources:
            return "(no relevant passages were found)"
        lines = []
        for i, s in enumerate(sources, 1):
            text = (s.get("context") or s.get("sentence") or "").strip()
            if text:
                lines.append(f"[{i}] {text}")
        return "\n".join(lines) if lines else "(no relevant passages were found)"

    def _build_prompt(
        self,
        query: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        triples: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        abbreviations: Optional[List[Tuple[str, str]]] = None,
        grounded: bool = True
    ) -> str:
        """Construct prompt grounded on source passages or knowledge graph facts."""
        use_sources = sources is not None
        facts_text = self.format_sources(sources) if use_sources else self.format_triples(triples or [])
        grounding_rule = (
            "4. Grounding: Base everything you say strictly on the numbered source passages below. "
            "The passages are verbatim extracts, so you may quote or closely paraphrase them, but "
            "never invent, assume, or extrapolate anything they do not state."
            if use_sources else
            "4. Grounding: Base everything you say strictly on the Knowledge Graph facts provided below. "
            "Never invent, assume, or extrapolate unmentioned facts."
        )
        not_grounded_rule = (
            ""
            if grounded else
            "6. IMPORTANT: Retrieved passages were judged NOT to contain a reliable answer to this "
            "question. Say plainly and briefly that you do not have that information. Do not speculate "
            "and do not fill the gap with general knowledge.\n\n"
        )

        abbrev_section = ""
        if abbreviations:
            abbrev_lines = [f"- {abbr}: {full}" for abbr, full in abbreviations]
            abbrev_section = "Relevant Abbreviations:\n" + "\n".join(abbrev_lines) + "\n\n"

        system_msg = (
            "You are Ghana Chat, a knowledgeable and helpful assistant for Ghana.\n\n"
            "Guidelines:\n"
            "1. Direct & Natural: Answer directly, clearly, and conversationally. Do NOT use formulaic preambles, artificial acknowledgments (never say 'Ghana Chat acknowledges...' or 'That is an important question...'), or robotic meta-talk. Get straight to the answer.\n"
            "2. Natural Narrative Flow: Weave the facts into smooth narrative prose and natural paragraphs. Do NOT use bullet points, numbered lists, or robotic phrasing like 'A person who...'. Do NOT mechanically recite raw relations (never say 'is a subclass of...', 'is an instance of...', 'is a facet of...', or 'falls under the umbrella of...'). Instead, explain what the person, organization, or concept actually does in real-world terms.\n"
            "3. Conciseness & Clean Finish: Keep your response concise, between 60 to 120 words. Always complete your final sentence cleanly—never trail off or leave a sentence cut off.\n"
            f"{grounding_rule}\n"
            "5. Missing Information: If the provided material does not contain the answer, simply and naturally state that you don't have that information.\n"
            f"{not_grounded_rule}"
            f"{abbrev_section}"
            f"{'Source Passages' if use_sources else 'Knowledge Graph Facts'}:\n{facts_text}"
        )

        full_messages = [{"role": "system", "content": system_msg}]

        # Append prior conversation turns if provided
        if history:
            for turn in history:
                if turn.get("role") in ("user", "assistant") and turn.get("content"):
                    full_messages.append({"role": turn["role"], "content": turn["content"]})

        # Append the latest user query
        full_messages.append({"role": "user", "content": query})

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            try:
                prompt = self.tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                for m in full_messages[1:]:
                    prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                prompt += "<|im_start|>assistant\n"
        else:
            prompt = f"System: {system_msg}\n\n"
            for m in full_messages[1:]:
                prompt += f"{m['role'].capitalize()}: {m['content']}\n\n"
            prompt += "Assistant:"

        # Disable think mode for instant, direct narrative generation on Qwen
        if "qwen" in self.model_id.lower():
            prompt += "<think>\n\n</think>\n"

        return prompt

    def generate(
        self,
        query: str,
        triples: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        abbreviations: Optional[List[Tuple[str, str]]] = None,
        max_new_tokens: int = 200,
        sources: Optional[List[Dict[str, Any]]] = None,
        grounded: bool = True
    ) -> Dict[str, Any]:
        """Generate a concise, natural response grounded on retrieved passages."""
        prompt = self._build_prompt(
            query=query,
            sources=sources,
            triples=triples,
            history=history,
            abbreviations=abbreviations,
            grounded=grounded
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        latency = time.time() - t0

        input_len = inputs["input_ids"].shape[1]
        gen_tokens = outputs[0][input_len:]
        raw_output = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Clean any think tags
        answer = raw_output
        if "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()
        if "<think>" in answer:
            answer = answer.replace("<think>", "").strip()

        tok_speed = len(gen_tokens) / max(latency, 1e-5)

        return {
            "answer": answer,
            "latency_s": round(latency, 3),
            "tokens_generated": len(gen_tokens),
            "tokens_per_sec": round(tok_speed, 1)
        }

    def generate_stream(
        self,
        query: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        triples: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        abbreviations: Optional[List[Tuple[str, str]]] = None,
        max_new_tokens: int = 200,
        grounded: bool = True
    ):
        """Stream generated response tokens one by one as they are produced."""
        from threading import Thread
        from transformers import TextIteratorStreamer

        prompt = self._build_prompt(
            query=query,
            sources=sources,
            triples=triples,
            history=history,
            abbreviations=abbreviations,
            grounded=grounded
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer
        )
        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()

        for chunk in streamer:
            clean_chunk = chunk.replace("<think>", "").replace("</think>", "")
            if clean_chunk:
                yield clean_chunk
