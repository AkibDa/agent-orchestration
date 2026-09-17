# encoder/transformer.py

import tensorflow as tf
from .tokenizer_pipeline import encode_text, generate_mask, char_lookup, MAX_LEN

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
  def __init__(self, max_len, vocab_size, embed_dim, **kwargs):
    super().__init__(**kwargs)
    self.supports_masking = True 
    
    self.token_emb = tf.keras.layers.Embedding(
      input_dim=vocab_size, 
      output_dim=embed_dim, 
      name="token_embedding"
    )
    self.pos_emb = tf.keras.layers.Embedding(
      input_dim=max_len, 
      output_dim=embed_dim, 
      name="position_embedding"
    )

  def call(self, x):
    seq_len = tf.shape(x)[-1]
    positions = tf.range(start=0, limit=seq_len, delta=1)
    return self.token_emb(x) + self.pos_emb(positions)

  def compute_mask(self, inputs, mask=None):
    # Dynamically generate the mask: True for chars, False for PAD (0)
    # Keras will automatically pass this to TransformerBlock.call(mask=...)
    return tf.math.not_equal(inputs, 0)

class TransformerBlock(tf.keras.layers.Layer):
  def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
    super().__init__(**kwargs)
    self.supports_masking = True
    
    # Explicitly divide embed_dim by num_heads to get the correct key_dim
    self.att = tf.keras.layers.MultiHeadAttention(
      num_heads=num_heads, 
      key_dim=embed_dim // num_heads 
    )
    self.ffn = tf.keras.Sequential([
      tf.keras.layers.Dense(ff_dim, activation="relu"),
      tf.keras.layers.Dense(embed_dim)
    ])
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
    self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs, training=False, mask=None):
    if mask is not None:
      mask = mask[:, tf.newaxis, :]
        
    attn_output = self.att(inputs, inputs, inputs, attention_mask=mask)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(inputs + attn_output)
    
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output, training=training)
    return self.layernorm2(out1 + ffn_output)

class ORCAEncoder(tf.keras.Model):
  def __init__(self, max_len, vocab_size, embed_dim, num_heads, ff_dim, num_layers, label_vocabs, **kwargs):
    super().__init__(**kwargs)
    self.supports_masking = True 
    
    self.embedding = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
    self.encoder_blocks = [
      TransformerBlock(embed_dim, num_heads, ff_dim) 
      for _ in range(num_layers)
    ]
    
    self.heads = {}
    for task_name, num_classes in label_vocabs.items():
        self.heads[task_name] = tf.keras.layers.Dense(num_classes, name=f"head_{task_name}")

  def call(self, inputs, training=False, mask=None):
    x = self.embedding(inputs)
    
    for block in self.encoder_blocks:
      # Explicitly route the padding mask into each attention block
      x = block(x, training=training, mask=mask)
        
    cls_rep = x[:, 0, :]
    return {task: head(cls_rep) for task, head in self.heads.items()}

if __name__ == "__main__":
  print("--- Complete ORCA Encoder Test ---")
  
  # 1. Configuration (V0 Baseline Values)
  EMBED_DIM = 128
  NUM_LAYERS = 2
  NUM_HEADS = 4
  FF_DIM = 256
  VOCAB_SIZE = len(char_lookup.get_vocabulary())
  
  # 2. Simulate task vocabularies from Day 1 dataset inspection
  # In Day 3, these numbers will be dynamically pulled from the dataset
  MOCK_LABEL_VOCABS = {
    "language": 2, 
    "intent": 2, 
    "location": 1, 
    "activity": 2, 
    "time_rel": 1, 
    "time_per": 2
  }
  
  # 3. Initialize Complete Model
  orca_model = ORCAEncoder(
    max_len=MAX_LEN, 
    vocab_size=VOCAB_SIZE, 
    embed_dim=EMBED_DIM, 
    num_heads=NUM_HEADS, 
    ff_dim=FF_DIM, 
    num_layers=NUM_LAYERS,
    label_vocabs=MOCK_LABEL_VOCABS
  )
  
  # 4. Data Pipeline
  sample_text = "tmrrw morning kochi area mein machli pakadne jaana safe hai?"
  batched_ids = tf.expand_dims(encode_text(sample_text), axis=0)          
  batched_mask = tf.expand_dims(generate_mask(batched_ids[0]), axis=0)  
  
  # 5. Forward Pass
  predictions = orca_model(batched_ids, training=False, mask=batched_mask)
  
  # 6. Verification
  print(f"Input text: {sample_text}\n")
  print("Output shapes (Batch Size, Classes):")
  for task, logits in predictions.items():
    print(f" - {task.ljust(10)}: {logits.shape}")
      
  print("\nTest Complete. We successfully converted character inputs into six separate intent predictions using a shared self-attention representation.")