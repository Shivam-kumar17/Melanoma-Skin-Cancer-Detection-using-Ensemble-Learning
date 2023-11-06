import tensorflow as tf
from tensorflow.keras import layers, models

def transformer_block(inputs, embed_dim, num_heads, ff_dim, dropout=0):
    # Multi-head self-attention
    x = layers.MultiHeadAttention(key_dim=embed_dim, num_heads=num_heads)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)

    # Feedforward network
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return layers.LayerNormalization(epsilon=1e-6)(x + inputs)

def ViT(input_shape, num_classes, num_transformer_blocks, embed_dim, num_heads, ff_dim, dropout=0):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Reshape((-1, x.shape[-1]))(x)

    for _ in range(num_transformer_blocks):
        output = transformer_block(x, embed_dim, num_heads, ff_dim, dropout)(x)

    return models.Model(inputs,output)

# Example usage :
# input_shape = (128, 128, 3)  # Replace with your image input shape
# num_classes = 2  # Number of classes for classification
# num_transformer_blocks = 4  # Number of transformer blocks
# embed_dim = 32  # Dimension of transformer embeddings
# num_heads = 8  # Number of attention heads in transformer blocks
# ff_dim = 32  # Feedforward dimension in transformer blocks

# vip_cnn_model = build_vip_cnn(input_shape, num_classes, num_transformer_blocks, embed_dim, num_heads, ff_dim)
# vip_cnn_model.summary()
