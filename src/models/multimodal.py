"""
multimodal.py
-------------------------
Base code to build the multimodal model combining CNN and MLP for composer classification.
"""

import tensorflow as tf
from keras import layers, models 

from .cnn import build_cnn_audio_model
from .mlp import build_feature_mlp


def build_multimodal_model(
    mel_bins: int,
    time_frames: int,
    num_engineered_features: int,
    num_classes: int,
    audio_embedding_dim: int = 256,
    feature_embedding_dim: int = 128,
    fusion_hidden_dim: int = 256,
    dropout_rate: float = 0.4
) -> tf.keras.Model:
    """
    Multimodal composer classification model combining:
    - CNN over mel-spectrograms (audio)
    - MLP over engineered audio/MIDI features

    Args:
    mel_bins : int
        Number of mel frequency bins (e.g., 128)
    time_frames : int
        Number of time frames in spectrogram
    num_engineered_features : int
        Length of engineered feature vector
    num_classes : int
        Number of output composer classes
    audio_embedding_dim : int
        Size of CNN audio embedding
    feature_embedding_dim : int
        Size of engineered-feature embedding
    fusion_hidden_dim : int
        Size of fusion dense layer
    dropout_rate : float
        Dropout rate for regularization

    Returns:
    tf.keras.Model
        Compiled multimodal Keras model
    """

    # ===== Audio CNN branch =====

    audio_cnn = build_cnn_audio_model(
        mel_bins=mel_bins,
        time_frames=time_frames,
        num_classes=audio_embedding_dim
    )

    # Remove softmax head → treat output as embedding
    audio_embedding = audio_cnn.output
    audio_embedding = layers.Dense(
        audio_embedding_dim,
        activation="relu",
        name="audio_embedding"
    )(audio_embedding)

    # ===== Engineered feature MLP branch =====

    feature_mlp = build_feature_mlp(
        num_features=num_engineered_features,
        embedding_dim=feature_embedding_dim
    )

    feature_embedding = feature_mlp.output

    # ===== Fusion =====

    fused = layers.Concatenate(name="fusion_concat")([
        audio_embedding,
        feature_embedding
    ])

    x = layers.Dense(
        fusion_hidden_dim,
        activation="relu"
    )(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(
        fusion_hidden_dim // 2,
        activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # ===== Output =====
    
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="composer_logits"
    )(x)

    model = models.Model(
        inputs=[
            audio_cnn.input,
            feature_mlp.input
        ],
        outputs=outputs,
        name="multimodal_composer_classifier"
    )

    return model
