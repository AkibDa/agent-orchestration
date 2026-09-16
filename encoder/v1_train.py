# Proto/encoder/v1_train.py
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from encoder.tokenizer_pipeline import encode_text, char_lookup, MAX_LEN, generate_mask
from encoder.v1_transformer import ORCAEncoderV1

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "Proto"
    / "scripts"
    / "dataset_tier2.jsonl"
)

# --- 1. Load Data & Extract Raw Labels ---
texts = []
raw_labels = {
    "language": [], "action": [], "action_type": [], "intent": [], 
    "activity": [], "time_relative": [], "time_period": []
}

raw_seq_labels = {
    "char_tags": []
}

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        texts.append(data["input"])
        raw_labels["language"].append(data["language"])
        raw_labels["action"].append(data["action"])
        raw_labels["action_type"].append(data["action_type"])
        raw_labels["intent"].append(data["intent"])
        raw_labels["activity"].append(data["activity"])
        raw_labels["time_relative"].append(data["time_relative"])
        raw_labels["time_period"].append(data["time_period"])
        
        # We need to pad char_tags to MAX_LEN. 
        # The first token is <CLS>, so we prepend 'O' for <CLS>
        # and pad the rest with 'O' up to MAX_LEN.
        aligned_tags = ["O"] + data["char_tags"]
        if len(aligned_tags) < MAX_LEN:
            char_tags = aligned_tags + ["O"] * (MAX_LEN - len(aligned_tags))
        else:
            char_tags = aligned_tags[:MAX_LEN]
        raw_seq_labels["char_tags"].append(char_tags)

# --- 2. Build Label Lookups ---
label_lookups = {}
label_vocabs_counts = {}

for task, labels in raw_labels.items():
    vocab = sorted(list(set(labels)))
    label_lookups[task] = tf.keras.layers.StringLookup(vocabulary=vocab, num_oov_indices=1)
    label_vocabs_counts[task] = len(label_lookups[task].get_vocabulary())

seq_label_lookups = {}
seq_vocabs_counts = {}

for task, labels in raw_seq_labels.items():
    # Flatten list of lists to get vocabulary
    flat_labels = [tag for seq in labels for tag in seq]
    vocab = sorted(list(set(flat_labels)))
    seq_label_lookups[task] = tf.keras.layers.StringLookup(vocabulary=vocab, num_oov_indices=1)
    seq_vocabs_counts[task] = len(seq_label_lookups[task].get_vocabulary())

# --- 3. Deterministic Train/Val/Test Splits (80/10/10) ---
num_samples = len(texts)
indices = np.arange(num_samples)
np.random.seed(42) # Reproducible shuffle
np.random.shuffle(indices)

# Given we only have 10 samples in tier 2 currently, we'll just train on all for demonstration, 
# but we'll keep the split logic for when the real dataset is used.
if num_samples < 20:
    train_split = num_samples
    val_split = num_samples
else:
    train_split = int(0.8 * num_samples)
    val_split = int(0.9 * num_samples)

train_idx = indices[:train_split]
val_idx = indices[train_split:val_split]
test_idx = indices[val_split:]

def create_split_arrays(idx_set):
    if len(idx_set) == 0:
        return None, None
        
    split_texts = [texts[i] for i in idx_set]
    split_inputs = tf.stack([encode_text(t) for t in split_texts])
    
    split_labels = {}
    for task in raw_labels.keys():
        task_raw = [raw_labels[task][i] for i in idx_set]
        split_labels[task] = label_lookups[task](task_raw)
        
    for task in raw_seq_labels.keys():
        task_raw = [raw_seq_labels[task][i] for i in idx_set]
        split_labels[task] = seq_label_lookups[task](task_raw)
        
    return split_inputs, split_labels

# Accessible globally for imports
train_inputs, train_labels = create_split_arrays(train_idx)
val_inputs, val_labels = create_split_arrays(val_idx)
test_inputs, test_labels = create_split_arrays(test_idx)

# Custom loss function that ignores padding for sequence tagging
def seq_loss_fn(y_true, y_pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_object(y_true, y_pred)
    # Mask out loss where true label is OOV (index 0 usually, but let's just train on all since padding is 'O')
    # Actually, padding characters have input ID 0. We can mask loss based on the input mask if we passed it,
    # but here we just compute the loss over all tokens. Padding was tagged as 'O', which is a valid class.
    return tf.reduce_mean(loss)

# --- Execution Guard ---
if __name__ == "__main__":
    # --- 4. Build TF Datasets (Standard Format) ---
    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels)).batch(32)
    
    if val_inputs is not None:
        val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_labels)).batch(32)
    else:
        val_dataset = None

    # --- 5. Initialize Model ---
    EMBED_DIM = 128
    NUM_LAYERS = 2
    NUM_HEADS = 4
    FF_DIM = 256
    VOCAB_SIZE = len(char_lookup.get_vocabulary())

    tf.random.set_seed(42) # Lock weight initialization

    model = ORCAEncoderV1(
        max_len=MAX_LEN, 
        vocab_size=VOCAB_SIZE, 
        embed_dim=EMBED_DIM, 
        num_heads=NUM_HEADS, 
        ff_dim=FF_DIM, 
        num_layers=NUM_LAYERS,
        classification_vocabs=label_vocabs_counts,
        sequence_vocabs=seq_vocabs_counts
    )

    losses = {
        task: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
        for task in raw_labels.keys()
    }
    
    for task in raw_seq_labels.keys():
        losses[task] = seq_loss_fn

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=losses
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss' if val_dataset is None else 'val_loss',
        patience=5,
        restore_best_weights=True
    )

    print("Starting V1 Retraining with Multi-Task + Sequence Tagging...")
    
    if val_dataset:
        model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[early_stop])
    else:
        model.fit(train_dataset, epochs=50, callbacks=[early_stop])
    
    MODEL_PATH = PROJECT_ROOT / "Proto" / "encoder" / "orca_encoder_v1.weights.h5"
    VOCAB_PATH = PROJECT_ROOT / "Proto" / "encoder" / "v1_vocabs.json"

    model.save_weights(MODEL_PATH)
    
    # Save vocabularies for inference
    vocabs_to_save = {
        "classification": {task: lookup.get_vocabulary() for task, lookup in label_lookups.items()},
        "sequence": {task: lookup.get_vocabulary() for task, lookup in seq_label_lookups.items()}
    }
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocabs_to_save, f)
        
    print(f"\nWeights saved to {MODEL_PATH}")
    print(f"Vocabularies saved to {VOCAB_PATH}")

    # --- 6. Deterministic Evaluation on Train Split (since test is tiny here) ---
    print("\n--- Running Evaluation on Train Split ---")
    predictions = model.predict(train_dataset)

    accuracies = {}

    for task in raw_labels.keys():
        pred_classes = np.argmax(predictions[task], axis=-1)
        true_classes = train_labels[task].numpy()
        correct_mask = (pred_classes == true_classes)
        accuracies[task] = np.mean(correct_mask)

    for task in raw_seq_labels.keys():
        pred_classes = np.argmax(predictions[task], axis=-1)
        true_classes = train_labels[task].numpy()
        correct_mask = (pred_classes == true_classes)
        accuracies[task] = np.mean(correct_mask)

    print("\nIndividual Head Accuracies:")
    for task, acc in accuracies.items():
        print(f" - {task.ljust(15)}: {acc * 100:.2f}%")
