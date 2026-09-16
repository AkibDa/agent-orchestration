import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from schemas.extraction import ExtractionResult, LocationItem, LocationRole, Action, ActionType, Intent, Language

from encoder.v1_transformer import ORCAEncoderV1
from encoder.tokenizer_pipeline import encode_text, char_lookup, MAX_LEN, decode_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = PROJECT_ROOT / "Proto" / "encoder" / "orca_encoder_v1.weights.h5"
VOCABS_PATH = PROJECT_ROOT / "Proto" / "encoder" / "v1_vocabs.json"

class V1Extractor:
    def __init__(self):
        self.model = None
        self.vocabs = None
        self.is_loaded = False

    def load(self):
        if self.is_loaded:
            return

        if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(VOCABS_PATH):
            import logging
            logging.getLogger(__name__).warning("V1 model or vocabs not found. Fast extraction disabled.")
            return

        with open(VOCABS_PATH, "r") as f:
            self.vocabs = json.load(f)

        c_vocabs = self.vocabs["classification"]
        s_vocabs = self.vocabs["sequence"]

        c_vocab_counts = {k: len(v) for k, v in c_vocabs.items()}
        s_vocab_counts = {k: len(v) for k, v in s_vocabs.items()}

        self.model = ORCAEncoderV1(
            max_len=MAX_LEN,
            vocab_size=len(char_lookup.get_vocabulary()),
            embed_dim=128,
            num_heads=4,
            ff_dim=256,
            num_layers=2,
            classification_vocabs=c_vocab_counts,
            sequence_vocabs=s_vocab_counts
        )

        # Build model by passing a dummy input
        dummy_input = tf.zeros((1, MAX_LEN), dtype=tf.int64)
        self.model(dummy_input)
        self.model.load_weights(WEIGHTS_PATH)
        self.is_loaded = True

    def _decode_bio_tags(self, query: str, char_tags: list[str]):
        """Parse character BIO tags into locations and count."""
        locations = []
        count_str = None
        
        current_entity = ""
        current_type = None

        query_chars = list(query)
        # Shift index by 1 because tag 0 corresponds to <CLS>
        for i, char in enumerate(query_chars):
            if i + 1 >= MAX_LEN:
                break
            
            tag = char_tags[i + 1]
            if tag.startswith("B-"):
                if current_entity and current_type:
                    if current_type == "COUNT":
                        count_str = current_entity
                    else:
                        locations.append({"text": current_entity, "role": current_type})
                
                current_type = tag[2:]
                current_entity = char
            elif tag.startswith("I-") and current_type == tag[2:]:
                current_entity += char
            else:
                if current_entity and current_type:
                    if current_type == "COUNT":
                        count_str = current_entity
                    else:
                        locations.append({"text": current_entity, "role": current_type})
                current_entity = ""
                current_type = None

        if current_entity and current_type:
            if current_type == "COUNT":
                count_str = current_entity
            else:
                locations.append({"text": current_entity, "role": current_type})

        return locations, count_str

    def _parse_count(self, count_str: str) -> int:
        if not count_str:
            return 1
        
        mapping = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5
        }
        
        c = count_str.strip().lower()
        if c.isdigit():
            return int(c)
        if c in mapping:
            return mapping[c]
        return 1

    def extract(self, query: str) -> tuple[ExtractionResult | None, float]:
        if not self.is_loaded:
            return None, 0.0

        encoded = encode_text(query)
        inputs = tf.expand_dims(encoded, axis=0)
        
        preds = self.model.predict(inputs, verbose=0)
        
        confidences = []
        result_dict = {}
        
        # Parse classification heads
        c_vocabs = self.vocabs["classification"]
        for task in c_vocabs.keys():
            logits = preds[task][0]
            probs = tf.nn.softmax(logits).numpy()
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]
            confidences.append(confidence)
            result_dict[task] = c_vocabs[task][best_idx]
            
        # Parse sequence heads
        s_vocabs = self.vocabs["sequence"]
        char_tags_logits = preds["char_tags"][0]
        char_tags_probs = tf.nn.softmax(char_tags_logits, axis=-1).numpy()
        char_tags_idx = np.argmax(char_tags_probs, axis=-1)
        
        char_tags_str = [s_vocabs["char_tags"][i] for i in char_tags_idx]
        print(f"DEBUG CHAR TAGS: {char_tags_str[:len(query)]}")
        
        locations, count_str = self._decode_bio_tags(query, char_tags_str)
        
        count_val = self._parse_count(count_str)
        
        overall_confidence = float(np.min(confidences))
        
        try:
            loc_items = [
                LocationItem(text=l["text"].strip(), role=LocationRole(l["role"])) 
                for l in locations if l["text"].strip()
            ]
            
            ext_res = ExtractionResult(
                action=Action(result_dict["action"]),
                action_type=ActionType(result_dict["action_type"]),
                intent=Intent(result_dict["intent"]),
                language=Language(result_dict["language"]),
                locations=loc_items,
                activity=result_dict["activity"],
                count=count_val,
                time_relative=result_dict["time_relative"],
                time_period=result_dict["time_period"]
            )
            return ext_res, overall_confidence
        except ValueError as e:
            import logging
            logging.getLogger(__name__).warning(f"V1 output normalization failed: {e}")
            return None, 0.0

v1_extractor_instance = V1Extractor()
v1_extractor_instance.load()
