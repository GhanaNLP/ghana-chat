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
        max_new_tokens: int = 150
    ) -> Dict[str, Any]:
        """Construct multi-turn prompt and generate response strictly adhering to the facts."""
        facts_text = self.format_triples(triples)

        system_msg = (
            "You are a helpful and factual knowledge assistant for Ghana. "
            "Answer the user's questions using ONLY the provided Knowledge Graph facts below. "
            "Do NOT invent, extrapolate, or hallucinate any facts not explicitly present in the knowledge graph. "
            "If the provided facts do not contain the answer, say 'Based on the available knowledge graph facts, I do not have information to answer that.'\n\n"
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
        response_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Clean thinking tags if any leaked
        if "</think>" in response_text:
            response_text = response_text.split("</think>")[-1].strip()

        tok_speed = len(gen_tokens) / max(latency, 1e-5)

        return {
            "answer": response_text,
            "latency_s": round(latency, 3),
            "tokens_generated": len(gen_tokens),
            "tokens_per_sec": round(tok_speed, 1)
        }
