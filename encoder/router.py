import sys
from pathlib import Path

PROTO_DIR = Path(__file__).resolve().parents[1]

if str(PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROTO_DIR))

import numpy as np
import tensorflow as tf
from pathlib import Path

from schemas.contracts import QueryPlan, GeoLocation, QueryTime
from encoder.tokenizer_pipeline import encode_text, char_lookup, MAX_LEN
from encoder.train import label_lookups
from encoder.transformer import ORCAEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "orca_encoder_v0.weights.h5"



GAZETTEER = {
    "kochi": (9.9312, 76.2673, "Kochi"),
    "chennai": (13.0827, 80.2707, "Chennai"),
    "mumbai": (19.0760, 72.8777, "Mumbai"),
    "goa": (15.2993, 74.1240, "Goa"),
    "vizag": (17.6868, 83.2185, "Vizag"),
    "puri": (19.8135, 85.8312, "Puri"),
    "digha": (21.6274, 87.5088, "Digha"),
    "coxs_bazar": (21.4272, 92.0058, "Coxs_Bazar"),
    "sundarbans": (21.9497, 88.9337, "Sundarbans"),
    "haldia": (22.0257, 88.0583, "Haldia"),
    "bakkhali": (21.2618, 88.2638, "Bakkhali"),
    "mandarmani": (21.6640, 87.6713, "Mandarmani"),
    "sagar island": (21.7960, 88.1384, "Sagar_Island"),
    "sagar": (21.7960, 88.1384, "Sagar_Island"),
    "kakdwip": (21.8764, 88.1878, "Kakdwip"),
    "fraserganj": (21.5833, 88.2500, "Fraserganj"),
}



def assemble_plan(query: str, labels: dict) -> QueryPlan:
    intent = labels.get("intent_label", "unknown")

    intent_agents = {
        "marine_safety": ["risk"],
        "pfz_search": ["pfz", "risk"],
        "marine_conditions": ["weather", "ocean", "risk"],
        "hazard_alert": ["weather", "risk"],
        "unknown": [],
    }

    loc_name = labels.get("location_label")

    loc = next(
        (
            GeoLocation(latitude=lat, longitude=lon, name=name)
            for key, (lat, lon, name) in GAZETTEER.items()
            if name == loc_name
        ),
        None,
    )

    return QueryPlan(
        query=query,
        intent=intent,
        language=labels.get("language_label", "en"),
        location=loc,
        time=QueryTime(
            relative=labels.get("time_rel_label"),
            period=labels.get("time_per_label"),
        ),
        activity=labels.get("activity_label"),
        agents=intent_agents.get(intent, []),
        dependencies=[],
        constraints=[
            {
                "type": "max_risk",
                "value": "MEDIUM",
            }
        ],
    )



def transformer_router(
    query: str,
    model,
    label_lookups,
    encode_text,
) -> QueryPlan:

    ids = tf.expand_dims(encode_text(query), axis=0)
    preds = model(ids, training=False)

    labels = {}

    for task in preds:
        probs = tf.nn.softmax(preds[task][0]).numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        label_str = label_lookups[task].get_vocabulary()[pred_idx]

        # Reject low-confidence intent predictions
        if task == "intent" and confidence < 0.6:
            label_str = "unknown"

        # Unknown location
        if task == "location" and label_str == "UNKNOWN_LOC":
            label_str = None

        # Tokenizer UNK fallback
        elif label_str == "[UNK]":
            label_str = "unknown" if task == "intent" else "none"

        labels[f"{task}_label"] = label_str

    return assemble_plan(query, labels)



def load_orca_model():

    EMBED_DIM = 128
    NUM_LAYERS = 2
    NUM_HEADS = 4
    FF_DIM = 256

    VOCAB_SIZE = len(char_lookup.get_vocabulary())

    label_vocabs_counts = {
        task: len(lookup.get_vocabulary())
        for task, lookup in label_lookups.items()
    }

    model = ORCAEncoder(
        max_len=MAX_LEN,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        label_vocabs=label_vocabs_counts,
    )

    dummy_input = tf.zeros(
        (1, MAX_LEN),
        dtype=tf.int32,
    )

    model(dummy_input, training=False)

    model.load_weights(MODEL_PATH)

    return model


_model = None


def get_model():
    global _model

    if _model is None:
        print("Loading ORCA Transformer...")
        _model = load_orca_model()
        print("ORCA Transformer loaded.")

    return _model



def route_query(query: str) -> QueryPlan:

    model = get_model()

    return transformer_router(
        query,
        model,
        label_lookups,
        encode_text,
    )



if __name__ == "__main__":

    test_queries = [
        "Kal subah Kochi ke paas fishing jaana safe hai?",
        "Chennai ke paas aaj nearest PFZ kaunsa hai?",
        "Digha ke paas aaj nearest PFZ kaunsa hai?",
        "Kolkata ke paas aaj nearest PFZ kaunsa hai?",
        "Kal subah randomtown mein fishing safe hai?",
    ]

    for query in test_queries:

        plan = route_query(query)

        print("\nQuery:", query)
        print("Language:", plan.language)
        print("Intent:", plan.intent)

        if plan.location:
            print("Location:", plan.location.name)
        else:
            print("Location: None")

        print("Agents:", plan.agents)