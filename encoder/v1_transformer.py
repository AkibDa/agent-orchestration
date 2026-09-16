import tensorflow as tf
from .transformer import TokenAndPositionEmbedding, TransformerBlock

class ORCAEncoderV1(tf.keras.Model):
    def __init__(self, max_len, vocab_size, embed_dim, num_heads, ff_dim, num_layers, classification_vocabs, sequence_vocabs, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True 
        
        self.embedding = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
        self.encoder_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) 
            for _ in range(num_layers)
        ]
        
        self.classification_heads = {}
        for task_name, num_classes in classification_vocabs.items():
            self.classification_heads[task_name] = tf.keras.layers.Dense(num_classes, name=f"head_{task_name}")

        self.sequence_heads = {}
        for task_name, num_classes in sequence_vocabs.items():
            # Sequence heads output predictions for every token in the sequence
            self.sequence_heads[task_name] = tf.keras.layers.Dense(num_classes, name=f"seq_head_{task_name}")

    def call(self, inputs, training=False, mask=None):
        x = self.embedding(inputs)
        
        for block in self.encoder_blocks:
            x = block(x, training=training, mask=mask)
            
        cls_rep = x[:, 0, :]
        
        outputs = {}
        for task, head in self.classification_heads.items():
            outputs[task] = head(cls_rep)
            
        for task, head in self.sequence_heads.items():
            outputs[task] = head(x)
            
        return outputs
