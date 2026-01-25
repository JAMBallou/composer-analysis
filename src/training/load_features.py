"""
load_data.py
----------
Load and prepare datasets for training.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd


def get_datasets(config, batch_size=None):
    """
    Loads audio and MIDI features, splits into train/val/test, and returns tf.data.Dataset objects.
    
    Args:
        config (dict): Configuration dictionary from load_experiment_config()
        batch_size (int): Batch size for datasets. If None, uses config["training"]["batch_size"]
    
    Returns:
        tuple: (train_ds, val_ds, test_ds, metadata)
            - train_ds, val_ds, test_ds: tf.data.Dataset objects
            - metadata: dict with num_classes, class_names, feature_dim, audio_shape
    """
    
    if batch_size is None:
        batch_size = config["training"].get("batch_size", 32)
    
    # Paths
    repo_root = Path(__file__).resolve().parents[3]
    features_dir = repo_root / "results" / "features"
    labels_path = features_dir / "labels.csv"
    
    # Load labels
    labels_df = pd.read_csv(labels_path)
    
    # Get unique composers and create label mapping
    composers = sorted(labels_df["composer"].unique())
    composer_to_idx = {c: i for i, c in enumerate(composers)}
    labels_df["label"] = labels_df["composer"].map(composer_to_idx)
    
    # Load audio and MIDI features
    audio_features = []
    midi_features = []
    labels = []
    
    for _, row in labels_df.iterrows():
        file_id = row["id"]
        
        # Load mel spectrogram (audio)
        mel_path = features_dir / "audio" / f"{file_id}_mel.npy"
        if mel_path.exists():
            mel = np.load(mel_path)
            audio_features.append(mel)
        else:
            continue
        
        # Load MIDI features
        midi_path = features_dir / "midi" / f"{file_id}_aux.npy"
        if midi_path.exists():
            midi = np.load(midi_path)
            midi_features.append(midi)
        else:
            continue
        
        labels.append(row["label"])
    
    # Convert to arrays
    X_audio = np.array(audio_features)  # Shape: (N, 128, 431, 1) or similar
    X_midi = np.array(midi_features)    # Shape: (N, feature_dim)
    y = np.array(labels)                # Shape: (N,)
    
    # Ensure audio has 4D shape (batch, height, width, channels)
    if len(X_audio.shape) == 3:
        X_audio = X_audio[..., np.newaxis]
    
    # Normalize MIDI features
    X_midi = (X_midi - X_midi.mean(axis=0)) / (X_midi.std(axis=0) + 1e-8)
    
    # Train/val/test split
    train_split = config["training"].get("train_split", 0.7)
    val_split = config["training"].get("val_split", 0.15)
    
    n_samples = len(X_audio)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Create datasets
    def create_dataset(x_audio, x_midi, y_labels):
        dataset = tf.data.Dataset.from_tensor_slices(((x_audio, x_midi), y_labels))
        dataset = dataset.shuffle(len(x_audio))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    train_ds = create_dataset(X_audio[train_idx], X_midi[train_idx], y[train_idx])
    val_ds = create_dataset(X_audio[val_idx], X_midi[val_idx], y[val_idx])
    test_ds = create_dataset(X_audio[test_idx], X_midi[test_idx], y[test_idx])
    
    # Metadata
    metadata = {
        "num_classes": len(composers),
        "class_names": composers,
        "feature_dim": X_midi.shape[1],
        "audio_shape": tuple(X_audio.shape[1:]),
    }
    
    return train_ds, val_ds, test_ds, metadata