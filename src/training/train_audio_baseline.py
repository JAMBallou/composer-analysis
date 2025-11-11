"""
train_audio_baseline.py
-----------------------
Train the CNN on Mel spectrograms only.
"""

from src.model.cnn_audio import build_audio_cnn
from src.utils.config import SAMPLE_RATE
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the CNN with standard callbacks.
    """
    callbacks = [
        ModelCheckpoint("results/checkpoints/audio_cnn_best.h5", save_best_only=True),
        EarlyStopping(patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks
    )
    return history
