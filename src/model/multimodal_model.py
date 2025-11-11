"""
multimodal_model.py
-------------------
Combines CNN (audio) + dense MIDI features into a single multimodal network.
"""

import tensorflow as tf
from keras import layers, models, Input

def build_multimodal_model(audio_shape, midi_shape, num_classes=14):
    """
    Audio CNN branch + MIDI dense branch → concatenation → softmax.
    """
    # Audio branch
    audio_input = Input(shape=audio_shape)
    x = layers.Conv2D(32, (3,3), activation='relu')(audio_input)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)

    # MIDI branch
    midi_input = Input(shape=(midi_shape,))
    y = layers.Dense(64, activation='relu')(midi_input)

    # Fusion
    combined = layers.concatenate([x, y])
    z = layers.Dense(128, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    output = layers.Dense(num_classes, activation='softmax')(z)

    model = models.Model(inputs=[audio_input, midi_input], outputs=output)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
