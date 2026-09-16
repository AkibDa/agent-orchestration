# Proto/models/evaluate_compositional.py
#
# V0 "freeze gate" test, per the adversarial error analysis:
#   1. Hand-authored minimal pairs that directly target the two known
#      weaknesses (optional `activity` / `time_per` slot absence).
#   2. A programmatic slot-isolation sweep: hold three slots fixed, vary
#      the fourth across all its template options, and check
#        (a) the varied task's prediction tracks the varied slot, and
#        (b) the OTHER five task predictions do not change at all.
#      (b) is the actual test of whether the shared [CLS] representation
#      encodes six independent pieces of information, or whether slots
#      are bleeding into each other (which is what the two known errors
#      look like: an absent `activity` mention still flips `time_per`
#      handling, etc).
#
# This does not train anything. It loads the frozen orca_v0 weights and
# only runs inference, exactly like evaluate_errors.py.

import random
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

from .tokenizer_pipeline import encode_text, char_lookup, MAX_LEN
from .transformer import ORCAEncoder
from .train import raw_labels, label_lookups
from ..data.encoder_data_gen import HINGLISH_SLOTS, BENGLISH_SLOTS

# --- 1. Model Initialization (identical to evaluate_errors.py) ---
EMBED_DIM = 128
NUM_LAYERS = 2
NUM_HEADS = 4
FF_DIM = 256
VOCAB_SIZE = len(char_lookup.get_vocabulary())

label_vocabs_counts = {task: len(lookup.get_vocabulary()) for task, lookup in label_lookups.items()}

model = ORCAEncoder(
  max_len=MAX_LEN,
  vocab_size=VOCAB_SIZE,
  embed_dim=EMBED_DIM,
  num_heads=NUM_HEADS,
  ff_dim=FF_DIM,
  num_layers=NUM_LAYERS,
  label_vocabs=label_vocabs_counts
)

dummy_input = tf.zeros((1, MAX_LEN), dtype=tf.int32)
model(dummy_input, training=False)
model.load_weights("/Users/skakibahammed/code_playground/sih_prep/orca_encoder_v0.weights.h5")

TASKS = list(raw_labels.keys())  # language, intent, location, activity, time_rel, time_per


def predict_frame(text):
  """Run one input through the model and return {task: predicted_label_str}."""
  ids = tf.expand_dims(encode_text(text), axis=0)
  preds = model(ids, training=False)
  return {
    task: label_lookups[task].get_vocabulary()[int(np.argmax(preds[task][0]))]
    for task in TASKS
  }


# =====================================================================
# Part 1: Hand-authored minimal pairs (from the adversarial error report)
# =====================================================================

MINIMAL_PAIRS = [
  {
    "name": "activity presence/absence (Digha, bn, original error)",
    "varying": "activity",
    "cases": [
      ("nirapod hbe? kal shokl digha-r kache",
       {"activity": "none", "time_per": "morning", "location": "Digha", "time_rel": "tomorrow"}),
      ("fishing korte nirapod hobe? kal shokl digha-r kache",
       {"activity": "fishing", "time_per": "morning", "location": "Digha", "time_rel": "tomorrow"}),
    ],
  },
  {
    "name": "activity presence/absence, fixed word order",
    "varying": "activity",
    "cases": [
      ("kal shokl digha-r kache nirapod hbe?",
       {"activity": "none", "time_per": "morning"}),
      ("kal shokl digha-r kache fishing korte nirapod hbe?",
       {"activity": "fishing", "time_per": "morning"}),
    ],
  },
  {
    "name": "time_period presence/absence",
    "varying": "time_per",
    "cases": [
      ("digha-r kache agamikal kemon hobe?",
       {"time_rel": "tomorrow", "time_per": "none"}),
      ("digha-r kache kal shokale kemon hobe?",
       {"time_rel": "tomorrow", "time_per": "morning"}),
    ],
  },
]


def run_minimal_pairs():
  print("=" * 70)
  print("PART 1: Hand-authored minimal pairs")
  print("=" * 70)
  all_ok = True
  for case in MINIMAL_PAIRS:
    print(f"\n[{case['name']}]  (isolates: {case['varying']})")
    for text, expected in case["cases"]:
      pred = predict_frame(text)
      case_ok = all(pred[k] == v for k, v in expected.items())
      all_ok = all_ok and case_ok
      print(f"  {'✓' if case_ok else '❌'} \"{text}\"")
      for k, v in expected.items():
        marker = "✓" if pred[k] == v else "❌"
        print(f"      {marker} {k}: expected={v!r} pred={pred[k]!r}")
  return all_ok


# =====================================================================
# Part 2: Programmatic slot-isolation sweep over data_gen templates
# =====================================================================

def expected_frame(combo, lang_code):
  """combo: {'time': (text, {'relative':..,'period':..}), 'location': (text, lbl),
              'activity': (text, lbl), 'intent': (text, lbl)}"""
  time_lbl = combo["time"][1]
  return {
    "language": lang_code,
    "intent": combo["intent"][1],
    "location": combo["location"][1],
    "activity": combo["activity"][1],
    "time_rel": time_lbl["relative"],
    "time_per": time_lbl["period"],
  }


SLOT_TO_TASKS = {
  "time": ("time_rel", "time_per"),
  "location": ("location",),
  "activity": ("activity",),
  "intent": ("intent",),
}


def slot_isolation_sweep(slots_dict, lang_code, num_bases=4, seed=7):
  """For num_bases random base combinations, vary each slot in turn across
  every option that slot's template offers, holding the other three fixed.
  Returns a list of per-sweep records for reporting."""
  rnd = random.Random(seed)
  slot_names = list(slots_dict.keys())  # time, location, activity, intent
  sweeps = []

  for _ in range(num_bases):
    base_combo = {name: rnd.choice(slots_dict[name]) for name in slot_names}

    for varying_slot in slot_names:
      target_tasks = SLOT_TO_TASKS[varying_slot]
      other_tasks = [t for t in TASKS if t not in target_tasks]

      variants = []
      for option in slots_dict[varying_slot]:
        combo = dict(base_combo)
        combo[varying_slot] = option
        components = [combo["time"][0], combo["location"][0], combo["activity"][0], combo["intent"][0]]
        text = " ".join(components)
        expected = expected_frame(combo, lang_code)
        pred = predict_frame(text)
        variants.append({"text": text, "expected": expected, "pred": pred})

      sweeps.append({
        "lang": lang_code,
        "varying_slot": varying_slot,
        "target_tasks": target_tasks,
        "other_tasks": other_tasks,
        "variants": variants,
      })

  return sweeps


def report_sweeps(sweeps):
  print("\n" + "=" * 70)
  print("PART 2: Programmatic slot-isolation sweep")
  print("=" * 70)

  n_target_correct = 0
  n_target_total = 0
  n_leaked = 0
  n_leak_checks = 0
  leak_examples = []

  for sweep in sweeps:
    for v in sweep["variants"]:
      for t in sweep["target_tasks"]:
        n_target_total += 1
        if v["pred"][t] == v["expected"][t]:
          n_target_correct += 1

    # Leakage check: for the "other" (should-be-untouched) tasks, do their
    # predictions actually stay constant while only varying_slot changes?
    base_other_preds = {t: sweep["variants"][0]["pred"][t] for t in sweep["other_tasks"]}
    for v in sweep["variants"][1:]:
      for t in sweep["other_tasks"]:
        n_leak_checks += 1
        if v["pred"][t] != base_other_preds[t]:
          n_leaked += 1
          leak_examples.append({
            "varying_slot": sweep["varying_slot"],
            "task_that_leaked": t,
            "text_a": sweep["variants"][0]["text"],
            "pred_a": base_other_preds[t],
            "text_b": v["text"],
            "pred_b": v["pred"][t],
          })

  print(f"\nTarget-task accuracy under slot variation: "
        f"{n_target_correct}/{n_target_total} "
        f"({100 * n_target_correct / max(1, n_target_total):.2f}%)")
  print(f"Cross-slot leakage (an unrelated task's prediction changed "
        f"when it shouldn't have): {n_leaked}/{n_leak_checks} "
        f"({100 * n_leaked / max(1, n_leak_checks):.2f}%)")

  if leak_examples:
    print("\n--- Leakage examples (compositionality failures) ---")
    for ex in leak_examples[:15]:
      print(f"\n  varying_slot={ex['varying_slot']}  leaked_task={ex['task_that_leaked']}")
      print(f"    A: \"{ex['text_a']}\"  ->  {ex['task_that_leaked']}={ex['pred_a']!r}")
      print(f"    B: \"{ex['text_b']}\"  ->  {ex['task_that_leaked']}={ex['pred_b']!r}")
    if len(leak_examples) > 15:
      print(f"\n  ... and {len(leak_examples) - 15} more leakage cases.")

  return n_target_correct, n_target_total, n_leaked, n_leak_checks


if __name__ == "__main__":
  pairs_ok = run_minimal_pairs()

  hi_sweeps = slot_isolation_sweep(HINGLISH_SLOTS, "hi-Latn", num_bases=4, seed=7)
  bn_sweeps = slot_isolation_sweep(BENGLISH_SLOTS, "bn-Latn", num_bases=4, seed=7)
  report_sweeps(hi_sweeps + bn_sweeps)

  print("\n" + "=" * 70)
  print("VERDICT")
  print("=" * 70)
  if pairs_ok:
    print("Minimal pairs: PASS — both known error slots resolve correctly in isolation.")
  else:
    print("Minimal pairs: FAIL — at least one of the two known slot-disentanglement "
          "errors still reproduces. Do not freeze V0 yet.")
  print("Check the leakage percentage above: if it's near 0%, the shared [CLS] "
        "representation is behaving compositionally and V0 is safe to freeze. "
        "If leakage is non-trivial, the six heads are still coupled through the "
        "shared representation and that's worth a short investigation (e.g. "
        "checking whether the heads dict should be a proper Keras tracked "
        "structure, as noted separately) before moving to the agent layer.")