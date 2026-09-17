# conversation/model.py

import time
import gc
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from schemas.extraction import ExtractionResult

CUDA_MODEL_ID = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"
MLX_MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

@dataclass
class LLMStats:
  stage: str
  raw_output: str
  prompt_tokens: int
  completion_tokens: int
  total_tokens: int
  latency_sec: float
  tokens_per_sec: float


def _cuda_available() -> bool:
  """Detect CUDA availability without importing MLX."""
  import platform
  if platform.system() == "Darwin":
      return False
  try:
    import torch
    return torch.cuda.is_available()
  except Exception:
    return False


class ConversationModel:
  def __init__(self):

    self.last_extract_stats: Optional[LLMStats] = None
    self.last_generate_stats: Optional[LLMStats] = None

    if _cuda_available():
      self.backend = "cuda"
      self._init_cuda()
    else:
      self.backend = "mlx"
      self._init_mlx()

    print(f"Conversation backend: {self.backend}")

  # ============================================================
  # CUDA INIT
  # ============================================================

  def _init_cuda(self):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading CUDA Qwen model ({CUDA_MODEL_ID})...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    self.tokenizer = AutoTokenizer.from_pretrained(CUDA_MODEL_ID)

    self.model = AutoModelForCausalLM.from_pretrained(
      CUDA_MODEL_ID,
      device_map="auto",
    )

    self.model.eval()
    print("CUDA Qwen model loaded successfully.")

  # ============================================================
  # MLX INIT
  # ============================================================

  def _init_mlx(self):
    # pyrefly: ignore [missing-import]
    from mlx_lm import load

    print(f"Loading MLX Qwen model ({MLX_MODEL_ID})...")
    res = load(MLX_MODEL_ID)
    self.model, self.tokenizer = res[0], res[1]
    self.system_prompt_caches = {}
    print("MLX Qwen model loaded successfully.")

  # ============================================================
  # MLX PROMPT CACHING (preserved from original)
  # ============================================================

  def _get_turn_cache(self, system_prompt: str, prompt: str):
    # pyrefly: ignore [missing-import]
    from mlx_lm.models.cache import make_prompt_cache
    # pyrefly: ignore [missing-import]
    import mlx.core as mx

    if system_prompt not in self.system_prompt_caches:
      sys_messages = [{"role": "system", "content": system_prompt}]
      sys_text = self.tokenizer.apply_chat_template(sys_messages, tokenize=False)
      sys_tokens = self.tokenizer.encode(sys_text)
      base_cache = make_prompt_cache(self.model)
      self.model(mx.array(sys_tokens)[None], cache=base_cache)
      self.system_prompt_caches[system_prompt] = (sys_text, base_cache)

    sys_text, base_cache = self.system_prompt_caches[system_prompt]

    cache_hit = bool(sys_text and prompt.startswith(sys_text))
    if cache_hit:
      user_prompt = prompt[len(sys_text):]
      turn_cache = make_prompt_cache(self.model)
      for c1, c2 in zip(base_cache, turn_cache):
        if hasattr(c1, "keys"):
          c2.keys = c1.keys
          c2.values = c1.values
          c2.offset = c1.offset
          c2.step = c1.step
    else:
      user_prompt = prompt
      turn_cache = None

    return user_prompt, turn_cache, cache_hit

  # ============================================================
  # STATS
  # ============================================================

  def reset_turn_stats(self):
    """Reset statistics for both stages at the start of a turn to prevent state leakage."""
    self.last_extract_stats = None
    self.last_generate_stats = None

  # ============================================================
  # EXTRACTION — DISPATCH
  # ============================================================

  def extract(self, system_prompt: str, user_message: str) -> ExtractionResult:
    self.reset_turn_stats()

    if self.backend == "cuda":
      return self._extract_cuda(system_prompt, user_message)
    return self._extract_mlx(system_prompt, user_message)

  # ============================================================
  # CUDA EXTRACTION
  # ============================================================

  def _extract_cuda(self, system_prompt: str, user_message: str) -> ExtractionResult:
    import torch

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

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
      outputs = self.model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=self.tokenizer.eos_token_id,
      )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    generated_ids = outputs[0][prompt_tokens:]
    if len(generated_ids) >= 200:
        import logging
        logging.getLogger(__name__).warning("Extraction truncated! Hit max_new_tokens=200 limit.")

    raw_output = self.tokenizer.decode(
      generated_ids,
      skip_special_tokens=True,
    ).strip()

    completion_tokens = len(generated_ids)
    total_tokens = prompt_tokens + completion_tokens
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0.0

    self.last_extract_stats = LLMStats(
      stage="INTAKE_EXTRACTION",
      raw_output=raw_output,
      prompt_tokens=prompt_tokens,
      completion_tokens=completion_tokens,
      total_tokens=total_tokens,
      latency_sec=elapsed,
      tokens_per_sec=tokens_per_sec,
    )

    return self._parse_extraction(raw_output)

  # ============================================================
  # MLX EXTRACTION
  # ============================================================

  def _extract_mlx(self, system_prompt: str, user_message: str) -> ExtractionResult:
    # pyrefly: ignore [missing-import]
    from mlx_lm import generate

    import logging
    import os
    logger = logging.getLogger(__name__)

    t_start = time.perf_counter()

    # 1. Prompt Construction
    t0_prompt = time.perf_counter()
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_message},
    ]
    prompt = self.tokenizer.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens_total = len(self.tokenizer.encode(prompt))
    t_prompt_ms = (time.perf_counter() - t0_prompt) * 1000.0

    # 2. Cache Lookup
    t0_cache = time.perf_counter()
    user_prompt, turn_cache, cache_hit = self._get_turn_cache(system_prompt, prompt)
    t_cache_ms = (time.perf_counter() - t0_cache) * 1000.0

    if os.environ.get("DEBUG_LLM") == "1":
      print(f"[CACHE] hit={cache_hit} user_prompt_len={len(user_prompt)}")

    # 3. Model Generation
    t0_gen = time.perf_counter()
    raw_output = generate(self.model, self.tokenizer, prompt=user_prompt, prompt_cache=turn_cache, max_tokens=200, verbose=False)
    t_gen_ms = (time.perf_counter() - t0_gen) * 1000.0

    completion_tokens = len(self.tokenizer.encode(raw_output))
    if completion_tokens >= 200:
        logger.warning("Extraction truncated! Hit max_tokens=200 limit.")

    # 4. JSON Parsing
    t0_parse = time.perf_counter()
    raw_output = raw_output.strip()
    result = self._parse_extraction(raw_output)
    t_parse_ms = (time.perf_counter() - t0_parse) * 1000.0

    t_total_ms = (time.perf_counter() - t_start) * 1000.0

    # Log the sub-stage profiling
    print(f"  [QWEN PROFILING] Total: {t_total_ms:.1f}ms | Prompt: {t_prompt_ms:.1f}ms | Cache: {t_cache_ms:.1f}ms | Generate: {t_gen_ms:.1f}ms | Parse: {t_parse_ms:.1f}ms")

    elapsed = t_total_ms / 1000.0
    total_tokens = prompt_tokens_total + completion_tokens
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0.0

    self.last_extract_stats = LLMStats(
        stage="INTAKE_EXTRACTION",
        raw_output=raw_output,
        prompt_tokens=prompt_tokens_total,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_sec=elapsed,
        tokens_per_sec=tokens_per_sec,
    )

    return result

  # ============================================================
  # SHARED EXTRACTION PARSER (robust fallback chain)
  # ============================================================

  def _parse_extraction(self, raw_output: str) -> ExtractionResult:
    """Shared parser used by both CUDA and MLX backends.

    Fallback chain:
      1. Strip markdown fences
      2. Repair missing closing brace
      3. model_validate_json (strict)
      4. json.loads → model_validate (lenient)
      5. Prune invalid fields → model_validate (field-tolerant)
      6. ExtractionResult() empty default
    """
    clean_json = raw_output.strip()

    # Strip markdown code fences
    if clean_json.startswith("```json"):
      clean_json = clean_json[7:]
    elif clean_json.startswith("```"):
      clean_json = clean_json[3:]
    if clean_json.endswith("```"):
      clean_json = clean_json[:-3]
    clean_json = clean_json.strip()

    # Repair truncated generation
    repaired = False
    if clean_json.startswith("{") and not clean_json.endswith("}"):
      repaired = True
      # Count open structures
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
      data = json.loads(clean_json)

      # Translate compact format if detected
      if any(k in data for k in ["a", "t", "i", "l", "act", "time", "c"]):
          full = {}
          if "a" in data: full["action"] = data["a"]
          if "t" in data: full["action_type"] = data["t"]
          if "i" in data: full["intent"] = data["i"]
          
          if "l" in data and isinstance(data["l"], list):
              locs = []
              role_map = {"REF": "REFERENCE", "TGT": "TARGET", "REG": "REGION"}
              for item in data["l"]:
                  if isinstance(item, list) and len(item) >= 2:
                      role_val = item[1].upper()
                      locs.append({"text": item[0], "role": role_map.get(role_val, role_val)})
                  elif isinstance(item, dict):
                      locs.append(item)
              full["locations"] = locs
              
          if "act" in data: full["activity"] = data["act"]
          if "time" in data: full["time_relative"] = data["time"]
          if "c" in data: full["count"] = data["c"]
          if "dist" in data: full["spatial_distance_km"] = data["dist"]
          if "dir" in data: full["spatial_direction"] = data["dir"]
          if "et" in data and data["et"] != "none": full["explanation_target"] = data["et"]
          
          for k in ["action", "action_type", "intent", "locations", "activity", "time_relative", "count", "spatial_distance_km", "spatial_direction", "explanation_target"]:
              if k in data and k not in full:
                  full[k] = data[k]
          data = full

      if "action" in data and isinstance(data["action"], str):
          data["action"] = data["action"].upper()
      if "action_type" in data and isinstance(data["action_type"], str):
          data["action_type"] = data["action_type"].upper()
      if "intent" in data and isinstance(data["intent"], str):
          data["intent"] = data["intent"].lower()

      res = ExtractionResult.model_validate(data)
      if repaired:
          logging.getLogger(__name__).warning(f"JSON was repaired successfully. Before: {raw_output}")
      return res
    except Exception as e:
              # Regex fallback extraction
              from schemas.extraction import Intent, Action, LocationRole, ActionType, LocationItem, Language

              logging.getLogger(__name__).error(f"JSON parsing failed completely for raw output: {raw_output}. Falling back to regex extraction.")

              # Try to extract location text via regex
              loc_match = re.search(r'"text"\s*:\s*"([^"]+)"', raw_output)
              intent_match = re.search(r'"intent"\s*:\s*"([^"]+)"', raw_output)
              action_match = re.search(r'"action"\s*:\s*"([^"]+)"', raw_output)
              activity_match = re.search(r'"activity"\s*:\s*"([^"]+)"', raw_output)

              loc_text = loc_match.group(1) if loc_match else None
              intent_val = intent_match.group(1) if intent_match else "unknown"
              action_val = action_match.group(1) if action_match else "ORCA_QUERY"

              if loc_text:
                  return ExtractionResult(
                      intent=Intent(intent_val) if intent_val in [i.value for i in Intent] else Intent.unknown,
                      action=Action(action_val) if action_val in [a.value for a in Action] else Action.ORCA_QUERY,
                      action_type=ActionType.LOCATE,
                      locations=[LocationItem(text=loc_text, role=LocationRole.REFERENCE)],
                      activity=activity_match.group(1) if activity_match else "none",
                      language=Language.en
                  )

              # Utter failure
              return ExtractionResult(
                  intent=Intent.unknown,
                  action=Action.CHAT,
                  chat_reply="I encountered an internal error parsing the query. Could you please rephrase your request?"
              )

  # ============================================================
  # TEXT GENERATION — DISPATCH
  # ============================================================

  def generate_text(
    self,
    system_prompt: str,
    user_message: str,
    max_new_tokens: int = 90,
    temperature: float = 0.4,
  ) -> str:
    if self.backend == "cuda":
      return self._generate_text_cuda(
        system_prompt, user_message, max_new_tokens, temperature,
      )
    return self._generate_text_mlx(
      system_prompt, user_message, max_new_tokens, temperature,
    )

  # ============================================================
  # CUDA TEXT GENERATION
  # ============================================================

  def _generate_text_cuda(
    self,
    system_prompt: str,
    user_message: str,
    max_new_tokens: int,
    temperature: float,
  ) -> str:
    import torch

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

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
      outputs = self.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        pad_token_id=self.tokenizer.eos_token_id,
      )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_tokens = outputs[0][prompt_tokens:]
    text = self.tokenizer.decode(
      new_tokens,
      skip_special_tokens=True,
    ).strip()

    completion_tokens = len(new_tokens)
    total_tokens = prompt_tokens + completion_tokens
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0.0

    self.last_generate_stats = LLMStats(
      stage="RESPONSE_SYNTHESIS",
      raw_output=text,
      prompt_tokens=prompt_tokens,
      completion_tokens=completion_tokens,
      total_tokens=total_tokens,
      latency_sec=elapsed,
      tokens_per_sec=tokens_per_sec,
    )

    return text

  # ============================================================
  # MLX TEXT GENERATION
  # ============================================================

  def _generate_text_mlx(
    self,
    system_prompt: str,
    user_message: str,
    max_new_tokens: int,
    temperature: float,
  ) -> str:
    # pyrefly: ignore [missing-import]
    from mlx_lm import generate
    # pyrefly: ignore [missing-import]
    from mlx_lm.sample_utils import make_sampler

    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_message},
    ]
    prompt = self.tokenizer.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
    )

    t0 = time.time()
    prompt_tokens = len(self.tokenizer.encode(prompt))

    user_prompt, turn_cache, cache_hit = self._get_turn_cache(system_prompt, prompt)

    import os
    if os.environ.get("DEBUG_LLM") == "1":
      print(f"[CACHE_GENERATE] hit={cache_hit} user_prompt_len={len(user_prompt)}")

    t0 = time.time()
    sampler = make_sampler(temp=max(temperature, 1e-5))
    text = generate(
      self.model,
      self.tokenizer,
      prompt=user_prompt,
      prompt_cache=turn_cache,
      max_tokens=max_new_tokens,
      sampler=sampler,
      verbose=True,
    )
    elapsed = time.time() - t0

    completion_tokens = len(self.tokenizer.encode(text))
    total_tokens = prompt_tokens + completion_tokens
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0.0

    self.last_generate_stats = LLMStats(
        stage="RESPONSE_SYNTHESIS",
        raw_output=text.strip(),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_sec=elapsed,
        tokens_per_sec=tokens_per_sec,
    )

    return text.strip()


_singleton = None

def get_conversation_model():
  global _singleton
  if _singleton is None:
    import os
    default_backend = "cuda" if _cuda_available() else "mlx"
    backend = os.getenv("ORCA_MODEL_BACKEND", default_backend).lower()
    if backend == "cuda":
      from .model_cuda import get_conversation_model_cuda
      _singleton = get_conversation_model_cuda()
    else:
      _singleton = ConversationModel()
  return _singleton
