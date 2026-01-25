"""
mlp.py
-------------------------
Base code to build the MLP to analyze non-spectrogram features for composer classification.
"""

import tensorflow as tf
from keras import layers, models


def build_feature_mlp(
    num_features: int,
    embedding_dim: int = 128,
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """
    MLP for processing aggregated audio/MIDI features.

    Args:
    num_features : int
        Length of the input feature vector
    embedding_dim : int
        Size of the learned feature embedding
    dropout_rate : float
        Dropout rate for regularization

    Returns:
    tf.keras.Model
        Keras model producing a feature embedding
    """

    inputs = layers.Input(
        shape=(num_features,),
        name="engineered_features"
    )

    x = inputs

    # ===== Dense blocks =====

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # ===== Embedding output =====
    embedding = layers.Dense(
        embedding_dim,
        activation="relu",
        name="feature_embedding"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=embedding,
        name="feature_mlp"
    )

    return model
