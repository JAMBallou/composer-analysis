"""
cnn_audio.py
------------
Defines CNN architecture for Mel spectrogram classification.
"""

import tensorflow as tf
from keras import layers, models

def build_audio_cnn(input_shape=(128, 431, 1), num_classes=14):
    """
    Builds a small but efficient CNN for composer classification.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
