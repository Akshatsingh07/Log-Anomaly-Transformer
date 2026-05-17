import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import keras

def build_log_transformer_model(vocab_size, sequence_length, num_heads=4, ff_dim=128, embedding_dim=64, num_layers=2):
    inputs = layers.Input(shape=(sequence_length,),name='token_ids')
    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    
    positions=tf.range(start=0, limit=sequence_length, delta=1)
    pos_layer=layers.Embedding(input_dim=sequence_length, output_dim=embedding_dim)(positions)
    x=embedding_layer+pos_layer
    x=layers.Dropout(0.1,name='Input_Dropout')(x)
        
    padding_mask = keras.ops.cast(
        keras.ops.not_equal(inputs, 0), dtype='float32'
    )                                           # (B, T)
    padding_mask = keras.ops.expand_dims(padding_mask, axis=-1)  # (B, T, 1)
    x = x * padding_mask  
    
    for i in range(num_layers):
        residual=x
        x_norm=layers.LayerNormalization(epsilon=1e-6,name=f'norm1_{i}')(x)
        attention_output=layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim//num_heads, name=f'mha_{i}')(x_norm, x_norm)
        attention_output=layers.Dropout(0.1)(attention_output)
        x=attention_output+residual
        
        residual=x
        x_norm=layers.LayerNormalization(epsilon=1e-6, name=f'norm2_{i}')(x)
        ff_output=layers.Dense(ff_dim, activation='gelu',name=f'ffn1_{i}')(x_norm)
        ff_output=layers.Dense(embedding_dim,name=f'ffn2_{i}')(ff_output) 
        ff_output=layers.Dropout(0.1)(ff_output)
        x=ff_output+residual
    x=layers.LayerNormalization(epsilon=1e-6,name=f'final_norm')(x)
        
    # x=layers.GlobalAveragePooling1D()(x)
        
    outputs=layers.Dense(vocab_size,activation='softmax',name='reconstruction')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs,name='LogTransformerAE')
    return model