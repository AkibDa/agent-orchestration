# Proto/encoder/train.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "processed"
    / "orca_training_encoder_data.jsonl"
)


import json
import numpy as np
import tensorflow as tf

from .tokenizer_pipeline import encode_text, char_lookup, MAX_LEN
from .transformer import ORCAEncoder

# --- 1. Load Data & Extract Raw Labels ---
texts = []
raw_labels = {
  "language": [], "intent": [], "location": [], 
  "activity": [], "time_rel": [], "time_per": []
}

with open(DATASET_PATH, "r", encoding="utf-8") as f :
  for line in f:
    data = json.loads(line)
    texts.append(data["input"])
    raw_labels["language"].append(data["language_label"])
    raw_labels["intent"].append(data["intent_label"])
    raw_labels["location"].append(data["location_label"])
    raw_labels["activity"].append(data["activity_label"])
    raw_labels["time_rel"].append(data["time_rel_label"])
    raw_labels["time_per"].append(data["time_per_label"])

# --- 2. Build Label Lookups ---
label_lookups = {}
label_vocabs_counts = {}

for task, labels in raw_labels.items():
  vocab = sorted(list(set(labels)))
  label_lookups[task] = tf.keras.layers.StringLookup(vocabulary=vocab, num_oov_indices=1)
  label_vocabs_counts[task] = len(label_lookups[task].get_vocabulary())

# --- 3. Deterministic Train/Val/Test Splits (80/10/10) ---
num_samples = len(texts)
indices = np.arange(num_samples)
np.random.seed(42) # Reproducible shuffle
np.random.shuffle(indices)

train_split = int(0.8 * num_samples)
val_split = int(0.9 * num_samples)

train_idx = indices[:train_split]
val_idx = indices[train_split:val_split]
test_idx = indices[val_split:]

def create_split_arrays(idx_set):
  split_texts = [texts[i] for i in idx_set]
  split_inputs = tf.stack([encode_text(t) for t in split_texts])
  
  split_labels = {}
  for task in raw_labels.keys():
    task_raw = [raw_labels[task][i] for i in idx_set]
    split_labels[task] = label_lookups[task](task_raw)
      
  return split_inputs, split_labels

# Accessible globally for imports
train_inputs, train_labels = create_split_arrays(train_idx)
val_inputs, val_labels = create_split_arrays(val_idx)
test_inputs, test_labels = create_split_arrays(test_idx)

# --- Execution Guard ---
if __name__ == "__main__":
  # --- 4. Build TF Datasets (Standard Format) ---
  train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels)).batch(32)
  val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_labels)).batch(32)
  test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels)).batch(32)

  # --- 5. Initialize Model ---
  EMBED_DIM = 128
  NUM_LAYERS = 2
  NUM_HEADS = 4
  FF_DIM = 256
  VOCAB_SIZE = len(char_lookup.get_vocabulary())

  tf.random.set_seed(42) # Lock weight initialization

  model = ORCAEncoder(
    max_len=MAX_LEN, 
    vocab_size=VOCAB_SIZE, 
    embed_dim=EMBED_DIM, 
    num_heads=NUM_HEADS, 
    ff_dim=FF_DIM, 
    num_layers=NUM_LAYERS,
    label_vocabs=label_vocabs_counts
  )

  losses = {
    task: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
    for task in raw_labels.keys()
  }

  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=losses
  )
  
  early_stop = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss',
      patience=5,
      restore_best_weights=True
    )

  print("Starting Retraining with Fixed Pipeline...")
  model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[early_stop])
  
  # Save parameters for evaluate_errors.py

  MODEL_PATH = PROJECT_ROOT / "orca_encoder_v0.weights.h5"

  model.save_weights(MODEL_PATH)
  print("\nWeights saved to {MODEL_PATH}")

  # --- 6. Deterministic Evaluation on Test Split ---
  print("\n--- Running Evaluation on Held-Out Test Split ---")
  test_predictions = model.predict(test_dataset)

  exact_match_mask = np.ones(len(test_idx), dtype=bool)
  accuracies = {}

  for task in raw_labels.keys():
    pred_classes = np.argmax(test_predictions[task], axis=-1)
    true_classes = test_labels[task].numpy()
    
    correct_mask = (pred_classes == true_classes)
    accuracies[task] = np.mean(correct_mask)
    exact_match_mask = exact_match_mask & correct_mask

  print("\nTest Set Individual Head Accuracies:")
  for task, acc in accuracies.items():
    print(f" - {task.ljust(10)}: {acc * 100:.2f}%")

  test_full_frame = np.mean(exact_match_mask) * 100
  print(f"\nTest Set Full-Frame Accuracy (All 6 Correct): {test_full_frame:.2f}%")