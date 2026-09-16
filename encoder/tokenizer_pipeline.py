import json
from pathlib import Path
import tensorflow as tf

# Project root: ORCA-team/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "processed"
    / "orca_training_encoder_data.jsonl"
)

# --- 1. Load Data & Define Vocabulary ---
texts = []

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line)["input"])

vocab_chars = ["<CLS>"] + sorted(list(set("".join(texts))))

# --- 2. Build Tokenizer Layers ---
char_lookup = tf.keras.layers.StringLookup(
    vocabulary=vocab_chars,
    mask_token="<PAD>",
    oov_token="[UNK]"
)

id_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_lookup.get_vocabulary(),
    invert=True,
    mask_token="<PAD>",
    oov_token="[UNK]"
)

# --- 3. Padding, Truncation & Masking Pipeline ---
MAX_LEN = 128
CLS_ID = char_lookup(tf.constant(["<CLS>"]))[0]

def encode_text(text):
  """Converts a string to a padded, fixed-length tensor of character IDs."""
  chars = tf.strings.unicode_split(text, input_encoding='UTF-8')
  ids = char_lookup(chars)
  
  # Prepend <CLS>
  ids = tf.concat([[CLS_ID], ids], axis=0)
  
  # Truncate to MAX_LEN
  ids = ids[:MAX_LEN]
  
  # Pad to MAX_LEN
  pad_len = tf.maximum(0, MAX_LEN - tf.shape(ids)[0])
  ids = tf.pad(ids, paddings=[[0, pad_len]])
  return ids

def decode_tokens(ids):
  """Converts a tensor of IDs back to a string, removing <CLS> and <PAD>."""
  chars = id_to_char(ids)
  valid_chars = tf.boolean_mask(chars, (chars != b'<PAD>') & (chars != b'<CLS>'))
  return tf.strings.reduce_join(valid_chars).numpy().decode('utf-8')

def generate_mask(ids):
  """Generates an attention mask (1 for real tokens, 0 for padding)."""
  return tf.cast(ids != 0, tf.int32)

# --- 4. Verification Suite ---
if __name__ == "__main__":
  print("--- Vocabulary Inspection ---")
  vocab = char_lookup.get_vocabulary()
  print(f"Index 0: {vocab[0]}")
  print(f"Index 1: {vocab[1]}")
  print(f"Index 2: {vocab[2]}")

  print("\n--- Test 2 & 3: Encoding & Decoding ---")
  sample = "nirapod hobe? fising"
  encoded = encode_text(sample)
  decoded = decode_tokens(encoded)
  
  print(f"Original: {sample}")
  print(f"Decoded:  {decoded}")
  assert sample == decoded, "Decoding failed to match original!"
  
  print("\n--- Test 4: Padding Structure ---")
  print(f"Padded Tensor Shape: {encoded.shape}") 
  print(f"First 5 IDs (<CLS> + chars): {encoded[:5].numpy()}") 
  print(f"Last 5 IDs (Padding):        {encoded[-5:].numpy()}") 

  print("\n--- Test 5: Attention Mask ---")
  mask = generate_mask(encoded)
  print(f"Mask Shape: {mask.shape}")
  
  char_len = len(sample) + 1
  print(f"Mask values for <CLS> and chars: {mask[:char_len].numpy()}") 
  print(f"Mask values for <PAD>:           {mask[char_len:char_len+5].numpy()}")
  
  print("\nAll pipeline tests passed successfully.")