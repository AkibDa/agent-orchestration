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

        inputs = {
            k: v.to(self.model.device)
            for k, v in inputs.items()
        }

        prompt_tokens = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature is not None,
        }

        if temperature is not None:
            generation_kwargs["temperature"] = max(
                temperature,
                1e-5,
            )

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

        return (
            text,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            elapsed,
            tps,
        )

    def extract(
        self,
        system_prompt: str,
        user_message: str,
    ):
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

        return self._parse_extraction(raw_output)

    def _parse_extraction(
        self,
        raw_output: str,
    ) -> ExtractionResult:
        """
        Parse Qwen extraction output.

        Qwen is instructed by prompts.py to use compact keys:

            a    -> action
            t    -> action_type
            i    -> intent
            l    -> locations
            act  -> activity
            time -> time_relative
            c    -> count
            dist -> spatial_distance_km
            dir  -> spatial_direction
            et   -> explanation_target
        """

        import json
        import logging
        import re

        clean_json = raw_output.strip()

        # ---------------------------------------------------------
        # Remove markdown code fences
        # ---------------------------------------------------------

        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        elif clean_json.startswith("```"):
            clean_json = clean_json[3:]

        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]

        clean_json = clean_json.strip()

        # ---------------------------------------------------------
        # Repair truncated JSON
        # ---------------------------------------------------------

        repaired = False

        if clean_json.startswith("{") and not clean_json.endswith("}"):
            repaired = True

            in_string = False
            escape = False
            open_braces = 0
            open_brackets = 0

            for c in clean_json:
                if escape:
                    escape = False
                    continue

                if c == "\\":
                    escape = True

                elif c == '"':
                    in_string = not in_string

                elif not in_string:
                    if c == "{":
                        open_braces += 1
                    elif c == "}":
                        open_braces -= 1
                    elif c == "[":
                        open_brackets += 1
                    elif c == "]":
                        open_brackets -= 1

            if in_string:
                clean_json += '"'

            clean_json = clean_json.rstrip(", \n")

            while open_brackets > 0:
                clean_json += "]"
                open_brackets -= 1

            while open_braces > 0:
                clean_json += "}"
                open_braces -= 1

            if not clean_json.endswith("}"):
                clean_json += "}"

        # ---------------------------------------------------------
        # Parse and normalize compact Qwen schema
        # ---------------------------------------------------------

        try:
            data = json.loads(clean_json)

            # Detect compact Qwen schema.
            if any(
                key in data
                for key in [
                    "a",
                    "t",
                    "i",
                    "l",
                    "act",
                    "time",
                    "c",
                ]
            ):
                full = {}

                # a -> action
                if "a" in data:
                    full["action"] = data["a"]

                # t -> action_type
                if "t" in data:
                    full["action_type"] = data["t"]

                # i -> intent
                if "i" in data:
                    full["intent"] = data["i"]

                # l -> locations
                if "l" in data and isinstance(data["l"], list):
                    locs = []

                    role_map = {
                        "REF": "REFERENCE",
                        "TGT": "TARGET",
                        "REG": "REGION",
                    }

                    for item in data["l"]:
                        if (
                            isinstance(item, list)
                            and len(item) >= 2
                        ):
                            role_val = str(item[1]).upper()

                            locs.append(
                                {
                                    "text": item[0],
                                    "role": role_map.get(
                                        role_val,
                                        role_val,
                                    ),
                                }
                            )

                        elif isinstance(item, dict):
                            locs.append(item)

                    full["locations"] = locs

                # act -> activity
                if "act" in data:
                    full["activity"] = data["act"]

                # time -> time_relative
                if "time" in data:
                    full["time_relative"] = data["time"]

                # c -> count
                if "c" in data:
                    full["count"] = data["c"]

                # dist -> spatial_distance_km
                if "dist" in data:
                    full["spatial_distance_km"] = data["dist"]

                # dir -> spatial_direction
                if "dir" in data:
                    full["spatial_direction"] = data["dir"]

                # et -> explanation_target
                if (
                    "et" in data
                    and data["et"] != "none"
                ):
                    full["explanation_target"] = data["et"]

                # Preserve already-expanded fields if present.
                for key in [
                    "action",
                    "action_type",
                    "intent",
                    "locations",
                    "activity",
                    "time_relative",
                    "count",
                    "spatial_distance_km",
                    "spatial_direction",
                    "explanation_target",
                ]:
                    if (
                        key in data
                        and key not in full
                    ):
                        full[key] = data[key]

                data = full

            # -----------------------------------------------------
            # Normalize enum values
            # -----------------------------------------------------

            if (
                "action" in data
                and isinstance(data["action"], str)
            ):
                data["action"] = data["action"].upper()

            if (
                "action_type" in data
                and isinstance(data["action_type"], str)
            ):
                data["action_type"] = (
                    data["action_type"].upper()
                )

                if data["action_type"] == "SAFE_ROUTE":
                    data["action_type"] = "SEARCH"

            if (
                "intent" in data
                and isinstance(data["intent"], str)
            ):
                data["intent"] = data["intent"].lower()

            # -----------------------------------------------------
            # Validate against ExtractionResult
            # -----------------------------------------------------

            result = ExtractionResult.model_validate(data)

            if repaired:
                logging.getLogger(__name__).warning(
                    "JSON was repaired successfully. "
                    f"Before: {raw_output}"
                )

            return result

        except Exception:
            # -----------------------------------------------------
            # Regex fallback
            # -----------------------------------------------------

            from schemas.extraction import (
                Intent,
                Action,
                LocationRole,
                ActionType,
                LocationItem,
                Language,
            )

            logging.getLogger(__name__).error(
                "JSON parsing failed completely for raw output: "
                f"{raw_output}. Falling back to regex extraction."
            )

            loc_match = re.search(
                r'"(text|l)"\s*:\s*(?:\[\[)?"([^"]+)"',
                raw_output,
            )

            intent_match = re.search(
                r'"(intent|i)"\s*:\s*"([^"]+)"',
                raw_output,
            )

            action_match = re.search(
                r'"(action|a)"\s*:\s*"([^"]+)"',
                raw_output,
            )

            activity_match = re.search(
                r'"(activity|act)"\s*:\s*"([^"]+)"',
                raw_output,
            )

            loc_text = (
                loc_match.group(2)
                if loc_match
                else None
            )

            intent_val = (
                intent_match.group(2)
                if intent_match
                else "unknown"
            )

            action_val = (
                action_match.group(2)
                if action_match
                else "ORCA_QUERY"
            )

            if loc_text:
                return ExtractionResult(
                    intent=(
                        Intent(intent_val)
                        if intent_val
                        in [i.value for i in Intent]
                        else Intent.unknown
                    ),
                    action=(
                        Action(action_val)
                        if action_val
                        in [a.value for a in Action]
                        else Action.ORCA_QUERY
                    ),
                    action_type=ActionType.LOCATE,
                    locations=[
                        LocationItem(
                            text=loc_text,
                            role=LocationRole.REFERENCE,
                        )
                    ],
                    activity=(
                        activity_match.group(1)
                        if activity_match
                        else "none"
                    ),
                    language=Language.en,
                )

            # Complete failure fallback.
            return ExtractionResult(
                intent=Intent.unknown,
                action=Action.CHAT,
                chat_reply=(
                    "I encountered an internal error parsing "
                    "the query. Could you please rephrase "
                    "your request?"
                ),
            )

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