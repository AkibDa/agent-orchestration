# Proto/models/evaluate_errors.py

import numpy as np
import tensorflow as tf

# Lock seeds to ensure prediction operations are deterministic
np.random.seed(42)
tf.random.set_seed(42)

from .tokenizer_pipeline import encode_text, char_lookup, MAX_LEN
from .transformer import ORCAEncoder
from .train import raw_labels, label_lookups, test_inputs, test_labels, test_idx, texts

# --- 1. Model Initialization ---
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

# --- Step A: Reproducibility Check ---
test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels)).batch(32)
test_predictions = model.predict(test_dataset, verbose=0)

exact_match_mask = np.ones(len(test_idx), dtype=bool)
for task in raw_labels.keys():
  pred_classes = np.argmax(test_predictions[task], axis=-1)
  exact_match_mask = exact_match_mask & (pred_classes == test_labels[task].numpy())

print("\n--- Step A: Reproducibility Check ---")
print(f"Normal Held-Out Full-Frame Accuracy: {np.mean(exact_match_mask) * 100:.2f}%")

# --- Adversarial Data Setup ---
ADVERSARIAL_DATA = [
  {"input": "tmrrw kchke pas fising sef h?", "language_label": "hi-Latn", "intent_label": "marine_safety", "location_label": "Kochi", "activity_label": "fishing", "time_rel_label": "tomorrow", "time_per_label": "none"},
  {"input": "nirapod hbe? kal shokl digha-r kache", "language_label": "bn-Latn", "intent_label": "marine_safety", "location_label": "Digha", "activity_label": "none", "time_rel_label": "tomorrow", "time_per_label": "morning"},
  {"input": "kal kch area mein fising jaana safe?", "language_label": "hi-Latn", "intent_label": "marine_safety", "location_label": "Kochi", "activity_label": "fishing", "time_rel_label": "tomorrow", "time_per_label": "none"},
  {"input": "tomrw morning goa ke paas phishing safe rahega?", "language_label": "hi-Latn", "intent_label": "marine_safety", "location_label": "Goa", "activity_label": "fishing", "time_rel_label": "tomorrow", "time_per_label": "morning"},
  {"input": "mumbai coast par kal sea kaisa h?", "language_label": "hi-Latn", "intent_label": "marine_conditions", "location_label": "Mumbai", "activity_label": "none", "time_rel_label": "tomorrow", "time_per_label": "none"},
  {"input": "mach dhorte jabo agamikal cox's bazar e kemon hobe", "language_label": "bn-Latn", "intent_label": "marine_conditions", "location_label": "Coxs_Bazar", "activity_label": "fishing", "time_rel_label": "tomorrow", "time_per_label": "none"},
  {"input": "sundarban area-y tmrw mrnng fshng krte nirapod ki", "language_label": "bn-Latn", "intent_label": "marine_safety", "location_label": "Sundarbans", "activity_label": "fishing", "time_rel_label": "tomorrow", "time_per_label": "morning"},
  {"input": "vizag sea mn kl shkl fising saf h?", "language_label": "hi-Latn", "intent_label": "marine_safety", "location_label": "Vizag", "activity_label": "fishing", "time_rel_label": "tomorrow", "time_per_label": "morning"},
  {"input": "puri-te kal machli pkdn sef h?", "language_label": "hi-Latn", "intent_label": "marine_safety", "location_label": "Puri", "activity_label": "fishing", "time_rel_label": "tomorrow", "time_per_label": "none"},
  {"input": "chennai area kal mrnng fishing kaisa h", "language_label": "hi-Latn", "intent_label": "marine_conditions", "location_label": "Chennai", "activity_label": "fishing", "time_rel_label": "tomorrow", "time_per_label": "morning"},
  {"input": "kl digha-r kache samudre jawa saf ki?", "language_label": "bn-Latn", "intent_label": "marine_safety", "location_label": "Digha", "activity_label": "none", "time_rel_label": "tomorrow", "time_per_label": "none"},
  {"input": "fisshing safe h kl vizag sea mn?", "language_label": "hi-Latn", "intent_label": "marine_safety", "location_label": "Vizag", "activity_label": "fishing", "time_rel_label": "tomorrow", "time_per_label": "none"},
  {"input": "tmrww morning chennai ke paas sea kaisa hai 😭", "language_label": "hi-Latn", "intent_label": "marine_conditions", "location_label": "Chennai", "activity_label": "none", "time_rel_label": "tomorrow", "time_per_label": "morning"},
  {"input": "sundarban area-y kal shokale mach dhrt kmn hb", "language_label": "bn-Latn", "intent_label": "marine_conditions", "location_label": "Sundarbans", "activity_label": "fishing", "time_rel_label": "tomorrow", "time_per_label": "morning"},
  {"input": "kl subh puri-te fising saf h", "language_label": "hi-Latn", "intent_label": "marine_safety", "location_label": "Puri", "activity_label": "fishing", "time_rel_label": "tomorrow", "time_per_label": "morning"}
]

adv_inputs = tf.stack([encode_text(d["input"]) for d in ADVERSARIAL_DATA])
adv_predictions = model.predict(adv_inputs, verbose=0)

# --- Step B: Per-task Adversarial Accuracy ---
print("\n--- Step B: Adversarial Performance by Task ---")
adv_exact_match = np.ones(len(ADVERSARIAL_DATA), dtype=bool)
adv_pred_dict = {}

for task in raw_labels.keys():
  pred_classes = np.argmax(adv_predictions[task], axis=-1)
  adv_pred_dict[task] = pred_classes
  true_classes = np.array([label_lookups[task](d[f"{task}_label"]) for d in ADVERSARIAL_DATA])
  
  task_correct = (pred_classes == true_classes)
  adv_exact_match = adv_exact_match & task_correct
  print(f" - {task.ljust(10)}: {np.mean(task_correct) * 100:.2f}%")

print(f"\nAdversarial Full-Frame: {np.mean(adv_exact_match) * 100:.2f}%")

# --- Step C: Individual Error Analysis ---
print("\n--- Step C: Individual Adversarial Error Analysis ---")
def get_label_string(task, index):
  return label_lookups[task].get_vocabulary()[index]

for i, data in enumerate(ADVERSARIAL_DATA):
  if not adv_exact_match[i]:
    print(f"\n❌ Adversarial Example #{i+1}")
    print(f"INPUT: {data['input']}")
    print("TRUE:")
    for task in raw_labels.keys():
      print(f"  {task} = {data[f'{task}_label']}")
    print("PRED:")
    for task in raw_labels.keys():
      pred_str = get_label_string(task, adv_pred_dict[task][i])
      true_str = data[f"{task}_label"]
      marker = "✓" if pred_str == true_str else "❌"
      print(f"  {marker} {task} = {pred_str}")