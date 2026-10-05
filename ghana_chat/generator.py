"""Knowledge Graph Grounded Response Generator using Qwen 2B."""

import time
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class GroundedGenerator:
    """Loads Qwen 2B and generates answers strictly grounded on retrieved triples."""

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
        if not triples:
            return "No knowledge graph facts found for this query."
        lines = [f"- {t['head']} | {t['relation']} | {t['tail']}" for t in triples]
        return "\n".join(lines)

    def generate(
        self,
        query: str,
        triples: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 400
    ) -> Dict[str, Any]:
        """Construct prompt, generate with MiniCPM 5 (1B) reasoning enabled by default, and return both thought and narrative answer."""
        facts_text = self.format_triples(triples)

        system_msg = (
            "You are Ghana Chat, a warm, friendly, and knowledgeable assistant for Ghana. "
            "Your goal is to explain facts about Ghana in an engaging, natural, and helpful conversation.\n\n"
            "Guidelines:\n"
            "1. Persona: Speak warmly, politely, and naturally like an engaging local guide.\n"
            "2. Format: Write strictly in flowing narrative prose and natural paragraphs. "
            "Do NOT use bullet points, numbered lists, or robotic phrasing like 'A person who...' or 'He is...'. "
            "Weave the facts smoothly into cohesive sentences.\n"
            "3. Grounding: Base everything you say strictly on the Knowledge Graph facts provided below. "
            "Never invent, assume, or extrapolate unmentioned facts.\n"
            "4. Missing Information: If the provided facts do not contain the answer, warmly and politely let the user know.\n\n"
            f"Knowledge Graph Facts:\n{facts_text}"
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

        # Extract reasoning thought process (<think>...</think>)
        reasoning = ""
        answer = raw_output

        if "<think>" in raw_output:
            if "</think>" in raw_output:
                parts = raw_output.split("</think>", 1)
                reasoning = parts[0].replace("<think>", "").strip()
                answer = parts[1].strip()
            else:
                # Still inside think block
                reasoning = raw_output.replace("<think>", "").strip()
                answer = ""
        elif "</think>" in raw_output:
            parts = raw_output.split("</think>", 1)
            reasoning = parts[0].strip()
            answer = parts[1].strip()

        # If answer is empty but reasoning has content, use reasoning as answer fallback
        if not answer and reasoning:
            answer = reasoning

        tok_speed = len(gen_tokens) / max(latency, 1e-5)

        return {
            "answer": answer,
            "reasoning": reasoning,
            "latency_s": round(latency, 3),
            "tokens_generated": len(gen_tokens),
            "tokens_per_sec": round(tok_speed, 1)
        }
