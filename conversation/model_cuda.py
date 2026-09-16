import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from schemas.extraction import ExtractionResult


MODEL_ID = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"


@dataclass
class LLMStats:
    stage: str
    raw_output: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_sec: float
    tokens_per_sec: float


class ConversationModelCUDA:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        print(f"Loading CUDA Qwen model ({MODEL_ID})...")
        print(f"GPU: {torch.cuda.get_device_name(0)}")


        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
        )

        self.model.eval()

        print("CUDA Qwen model loaded successfully.")

        self.backend = "cuda"
        self.last_extract_stats: Optional[LLMStats] = None
        self.last_generate_stats: Optional[LLMStats] = None

    def reset_turn_stats(self):
        self.last_extract_stats = None
        self.last_generate_stats = None

    def _generate(
        self,
        system_prompt: str,
        user_message: str,
        max_new_tokens: int,
        temperature: float | None = None,
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        prompt_tokens = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature is not None,
        }

        if temperature is not None:
            generation_kwargs["temperature"] = max(temperature, 1e-5)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0

        generated_ids = outputs[0][prompt_tokens:]
        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        completion_tokens = len(generated_ids)
        total_tokens = prompt_tokens + completion_tokens

        tps = (
            completion_tokens / elapsed
            if elapsed > 0
            else 0.0
        )

        return text, prompt_tokens, completion_tokens, total_tokens, elapsed, tps

    def extract(self, system_prompt: str, user_message: str):
        self.reset_turn_stats()

        (
            raw_output,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            elapsed,
            tps,
        ) = self._generate(
            system_prompt,
            user_message,
            max_new_tokens=200,
            temperature=None,
        )

        self.last_extract_stats = LLMStats(
            stage="INTAKE_EXTRACTION",
            raw_output=raw_output.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_sec=elapsed,
            tokens_per_sec=tps,
        )

        clean_json = raw_output.strip()

        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        elif clean_json.startswith("```"):
            clean_json = clean_json[3:]

        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]

        clean_json = clean_json.strip()

        import json
        
        if clean_json.startswith("{") and not clean_json.endswith("}"):
            in_string = False
            escape = False
            open_braces = 0
            open_brackets = 0
            for c in clean_json:
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                elif c == '"':
                    in_string = not in_string
                elif not in_string:
                    if c == '{': open_braces += 1
                    elif c == '}': open_braces -= 1
                    elif c == '[': open_brackets += 1
                    elif c == ']': open_brackets -= 1
            if in_string:
                clean_json += '"'
            clean_json = clean_json.rstrip(', \n')
            while open_brackets > 0:
                clean_json += "]"
                open_brackets -= 1
            while open_braces > 0:
                clean_json += "}"
                open_braces -= 1
            if not clean_json.endswith("}"):
                clean_json += "}"

        try:
            return ExtractionResult.model_validate_json(clean_json)
        except Exception:
            try:
                data = json.loads(clean_json)
                try:
                    return ExtractionResult.model_validate(data)
                except Exception as val_err:
                    full = data
                    if "act" in data: full["activity"] = data["act"]
                    if "time" in data: full["time_relative"] = data["time"]
                    if "c" in data: full["count"] = data["c"]
                    if "dist" in data: full["spatial_distance_km"] = data["dist"]
                    if "dir" in data: full["spatial_direction"] = data["dir"]
                    if "et" in data and data["et"] != "none": full["explanation_target"] = data["et"]

                    if hasattr(val_err, 'errors'):
                        for err in val_err.errors():
                            loc = err.get('loc', [])
                            if loc and loc[0] in full:
                                del full[loc[0]]
                        return ExtractionResult.model_validate(full)
                    else:
                        return ExtractionResult()
            except Exception:
                return ExtractionResult()

    def generate_text(
        self,
        system_prompt: str,
        user_message: str,
        max_new_tokens: int = 90,
        temperature: float = 0.4,
    ) -> str:

        (
            text,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            elapsed,
            tps,
        ) = self._generate(
            system_prompt,
            user_message,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        self.last_generate_stats = LLMStats(
            stage="RESPONSE_SYNTHESIS",
            raw_output=text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_sec=elapsed,
            tokens_per_sec=tps,
        )

        return text.strip()


_singleton: ConversationModelCUDA | None = None


def get_conversation_model_cuda():
    global _singleton

    if _singleton is None:
        _singleton = ConversationModelCUDA()

    return _singleton