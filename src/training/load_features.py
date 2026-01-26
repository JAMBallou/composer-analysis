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
    repo_root = Path(__file__).resolve().parents[2]
    features_dir = repo_root / "results" / "features"
    labels_path = features_dir / "labels.csv"
    
    # Load labels with UTF-8 encoding
    labels_df = pd.read_csv(labels_path, encoding='utf-8')
    print(f"Total samples in labels.csv: {len(labels_df)}")
    print(f"Unique composers: {labels_df['composer'].unique()[:5]}")  # Show first 5
    
    # Filter composers based on config if specified
    if "dataset" in config and "composers" in config["dataset"]:
        config_composers = config["dataset"]["composers"]
        print(f"Config composers to filter: {config_composers}")
        labels_df = labels_df[labels_df["composer"].isin(config_composers)]
        print(f"Filtered to composers: {labels_df['composer'].unique()}")
        print(f"Samples after filtering: {len(labels_df)}")
    
    # Get unique composers and create label mapping
    composers = sorted(labels_df["composer"].unique())
    print(f"Composers in dataset: {composers}")
    composer_to_idx = {c: i for i, c in enumerate(composers)}
    labels_df["label"] = labels_df["composer"].map(composer_to_idx)
    
    # Load audio and MIDI features
    audio_features = []
    midi_features = []
    labels = []
    missing_count = 0
    
    for idx, (_, row) in enumerate(labels_df.iterrows()):
        file_id = str(row["id"]).zfill(4)  # Convert to 4-digit zero-padded string
        
        # Load mel spectrogram (audio)
        mel_path = features_dir / "audio" / f"{file_id}_mel.npy"
        midi_path = features_dir / "midi" / f"{file_id}_midi.npy"
        
        # Only add sample if BOTH files exist
        if not mel_path.exists() or not midi_path.exists():
            missing_count += 1
            if missing_count <= 3:  # Print first 3 missing files
                print(f"Warning: Missing files for {file_id}")
                print(f"  Mel exists: {mel_path.exists()} at {mel_path}")
                print(f"  MIDI exists: {midi_path.exists()} at {midi_path}")
            continue
        
        try:
            mel = np.load(mel_path)
            audio_features.append(mel)
            
            midi = np.load(midi_path)
            midi_features.append(midi)
            
            labels.append(row["label"])
        except Exception as e:
            print(f"Warning: Could not load files for {file_id}: {e}")
            continue
    
    if len(audio_features) == 0:
        print(f"ERROR: No valid audio/MIDI feature pairs found!")
        print(f"Total samples checked: {len(labels_df)}")
        print(f"Missing samples: {missing_count}")
        raise ValueError(f"No valid audio/MIDI feature pairs found in {features_dir}")
    
    # Convert to arrays
    X_audio = np.array(audio_features)  # Shape: (N, 128, 431, 1) or similar
    X_midi = np.array(midi_features)    # Shape: (N, feature_dim)
    y = np.array(labels)                # Shape: (N,)
    
    print(f"\n=== Data Shapes Before Processing ===")
    print(f"X_audio shape: {X_audio.shape}")
    print(f"X_midi shape: {X_midi.shape}")
    print(f"y shape: {y.shape}")
    print(f"y unique values: {np.unique(y)}")
    print(f"y value counts: {np.bincount(y)}")
    
    # Ensure audio has 4D shape (batch, height, width, channels)
    if len(X_audio.shape) == 3:
        X_audio = X_audio[..., np.newaxis]
        print(f"Added channel dimension. X_audio shape: {X_audio.shape}")
    
    # Normalize audio features
    X_audio = X_audio.astype(np.float32)
    X_audio = (X_audio - X_audio.mean()) / (X_audio.std() + 1e-8)
    print(f"Normalized X_audio - mean: {X_audio.mean():.4f}, std: {X_audio.std():.4f}")
    
    # Normalize MIDI features (handle empty case)
    if X_midi.size > 0:
        X_midi = X_midi.astype(np.float32)
        # Handle per-feature normalization
        if len(X_midi.shape) == 1:
            X_midi = X_midi.reshape(-1, 1)
        X_midi = (X_midi - X_midi.mean(axis=0)) / (X_midi.std(axis=0) + 1e-8)
        print(f"Normalized X_midi shape: {X_midi.shape}")
    
    print(f"=== Data Shapes After Processing ===")
    print(f"X_audio shape: {X_audio.shape}")
    print(f"X_midi shape: {X_midi.shape}")
    
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