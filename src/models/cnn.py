"""
cnn.py
-------------------------
Base code to build the CNN to analyze mel-spectrogram audio features for composer classification.
"""

import tensorflow as tf
from keras import layers, models


def build_cnn_audio_model(
    mel_bins: int = 128,
    time_frames: int = 2580,
    num_classes: int = 2,
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """
    CNN model for mel-spectrogram-based composer classification.

    Args:
    mel_bins (int): Number of mel frequency bins (default: 128)
    time_frames (int): Number of time frames (default: 2580)
    num_classes (int): Number of output classes (default: 2 for binary; 5 for subset; 14 for full task)
    dropout_rate (float): Dropout rate for regularization

    Returns:
    tf.keras.Model: Compiled Keras model
    """
    
    # ===== Input =====
    inputs = layers.Input(
        shape=(mel_bins, time_frames, 1),
        name="mel_spectrogram"
    )

    x = inputs

    # ===== Convolutional Blocks =====

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Optional deeper block (useful for Trial 3)
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)


    # ===== Global pooling =====
    x = layers.GlobalAveragePooling2D()(x)


    # ===== Dense classification head =====
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)


    # ===== Output =====
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="cnn_output"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn_mel_spectrogram_classifier"
    )

    return model
