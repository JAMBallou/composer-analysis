"""
load_data.py
----------
Load and prepare datasets for training.

Updated: Added temporal 3-segment data loading for new architecture.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from ..utils.losses import compute_class_weights_balanced, compute_class_weights_inverse_frequency


def resolve_features_dir(repo_root, config):
    """Resolve features directory with a fallback to outputs/features."""

    config_dir = config.get("dataset", {}).get("features_dir")
    if config_dir:
        return Path(config_dir)

    candidates = [repo_root / "results" / "features", repo_root / "outputs" / "features"]
    for candidate in candidates:
        if (candidate / "labels.csv").exists():
            return candidate

    return candidates[0]


def get_temporal_datasets(config, batch_size=None):
    """
    Loads 3-segment temporal features (start, middle, end) for audio and MIDI.
    
    Returns datasets formatted for temporal multimodal model:
    - Spectrograms: 3 inputs (start, middle, end)
    - Numerical features: 3 inputs (aux+MIDI for start, middle, end)
    
    Args:
        config (dict): Configuration dictionary from load_experiment_config()
        batch_size (int): Batch size for datasets. If None, uses config["training"]["batch_size"]
    
    Returns:
        tuple: (train_ds, val_ds, test_ds, metadata)
    """
    
    if batch_size is None:
        batch_size = config["training"].get("batch_size", 32)
    
    # Paths
    repo_root = Path(__file__).resolve().parents[2]
    features_dir = resolve_features_dir(repo_root, config)
    labels_path = features_dir / "labels.csv"
    
    # Load labels
    labels_df = pd.read_csv(labels_path, encoding='utf-8')
    print(f"Total samples in labels.csv: {len(labels_df)}")
    
    # Filter composers if specified
    if "dataset" in config and "composers" in config["dataset"]:
        config_composers = config["dataset"]["composers"]
        labels_df = labels_df[labels_df["composer"].isin(config_composers)]
        print(f"Filtered to {len(labels_df)} samples")
    
    # Create label mapping
    composers = sorted(labels_df["composer"].unique())
    print(f"Composers: {composers}")
    composer_to_idx = {c: i for i, c in enumerate(composers)}
    labels_df["label"] = labels_df["composer"].map(composer_to_idx)
    
    # Load 3-segment features
    spec_start_list, spec_middle_list, spec_end_list = [], [], []
    aux_start_list, aux_middle_list, aux_end_list = [], [], []
    midi_start_list, midi_middle_list, midi_end_list = [], [], []
    labels = []
    missing_count = 0
    
    segments = ["start", "middle", "end"]
    
    for idx, (_, row) in enumerate(labels_df.iterrows()):
        file_id = str(row["id"]).zfill(4)
        
        # Check all required files exist
        required_files = []
        for seg in segments:
            required_files.append(features_dir / "audio" / f"{file_id}_mel_{seg}.npy")
            required_files.append(features_dir / "audio" / f"{file_id}_aux_{seg}.npy")
            required_files.append(features_dir / "midi" / f"{file_id}_midi_{seg}.npy")
        
        if not all(f.exists() for f in required_files):
            missing_count += 1
            if missing_count <= 3:
                print(f"Warning: Missing files for {file_id}")
            continue
        
        try:
            # Load spectrograms
            spec_start = np.load(features_dir / "audio" / f"{file_id}_mel_start.npy")
            spec_middle = np.load(features_dir / "audio" / f"{file_id}_mel_middle.npy")
            spec_end = np.load(features_dir / "audio" / f"{file_id}_mel_end.npy")
            
            # Load auxiliary features
            aux_start = np.load(features_dir / "audio" / f"{file_id}_aux_start.npy")
            aux_middle = np.load(features_dir / "audio" / f"{file_id}_aux_middle.npy")
            aux_end = np.load(features_dir / "audio" / f"{file_id}_aux_end.npy")
            
            # Load MIDI features
            midi_start = np.load(features_dir / "midi" / f"{file_id}_midi_start.npy")
            midi_middle = np.load(features_dir / "midi" / f"{file_id}_midi_middle.npy")
            midi_end = np.load(features_dir / "midi" / f"{file_id}_midi_end.npy")
            
            # Append to lists
            spec_start_list.append(spec_start)
            spec_middle_list.append(spec_middle)
            spec_end_list.append(spec_end)
            
            aux_start_list.append(aux_start)
            aux_middle_list.append(aux_middle)
            aux_end_list.append(aux_end)
            
            midi_start_list.append(midi_start)
            midi_middle_list.append(midi_middle)
            midi_end_list.append(midi_end)
            
            labels.append(row["label"])
            
        except Exception as e:
            print(f"Error loading {file_id}: {e}")
            continue
    
    if len(labels) == 0:
        raise ValueError(
            "No valid temporal features found. Check that the feature files exist in the "
            "expected outputs/features structure and match the temporal naming scheme."
        )
    
    # Convert to arrays
    X_spec_start = np.array(spec_start_list)
    X_spec_middle = np.array(spec_middle_list)
    X_spec_end = np.array(spec_end_list)
    
    X_aux_start = np.array(aux_start_list)
    X_aux_middle = np.array(aux_middle_list)
    X_aux_end = np.array(aux_end_list)
    
    X_midi_start = np.array(midi_start_list)
    X_midi_middle = np.array(midi_middle_list)
    X_midi_end = np.array(midi_end_list)
    
    y = np.array(labels, dtype=np.int32)
    
    print(f"\n=== Temporal Data Shapes ===")
    print(f"Spectrograms: {X_spec_start.shape}")
    print(f"Aux features: {X_aux_start.shape}")
    print(f"MIDI features: {X_midi_start.shape}")
    print(f"Labels: {y.shape}, unique: {np.unique(y)}")
    
    # Add channel dimension to spectrograms if needed
    if len(X_spec_start.shape) == 3:
        X_spec_start = X_spec_start[..., np.newaxis]
        X_spec_middle = X_spec_middle[..., np.newaxis]
        X_spec_end = X_spec_end[..., np.newaxis]
    
    # Normalize spectrograms (in-place to avoid loop reassignment bug)
    X_spec_start = X_spec_start.astype(np.float32)
    X_spec_middle = X_spec_middle.astype(np.float32)
    X_spec_end = X_spec_end.astype(np.float32)
    
    mean_start, std_start = X_spec_start.mean(), X_spec_start.std()
    X_spec_start = (X_spec_start - mean_start) / (std_start + 1e-8)
    
    mean_middle, std_middle = X_spec_middle.mean(), X_spec_middle.std()
    X_spec_middle = (X_spec_middle - mean_middle) / (std_middle + 1e-8)
    
    mean_end, std_end = X_spec_end.mean(), X_spec_end.std()
    X_spec_end = (X_spec_end - mean_end) / (std_end + 1e-8)
    
    # Normalize numerical features (in-place)
    X_aux_start = X_aux_start.astype(np.float32)
    X_aux_start = (X_aux_start - X_aux_start.mean(axis=0)) / (X_aux_start.std(axis=0) + 1e-8)
    
    X_aux_middle = X_aux_middle.astype(np.float32)
    X_aux_middle = (X_aux_middle - X_aux_middle.mean(axis=0)) / (X_aux_middle.std(axis=0) + 1e-8)
    
    X_aux_end = X_aux_end.astype(np.float32)
    X_aux_end = (X_aux_end - X_aux_end.mean(axis=0)) / (X_aux_end.std(axis=0) + 1e-8)
    
    X_midi_start = X_midi_start.astype(np.float32)
    X_midi_start = (X_midi_start - X_midi_start.mean(axis=0)) / (X_midi_start.std(axis=0) + 1e-8)
    
    X_midi_middle = X_midi_middle.astype(np.float32)
    X_midi_middle = (X_midi_middle - X_midi_middle.mean(axis=0)) / (X_midi_middle.std(axis=0) + 1e-8)
    
    X_midi_end = X_midi_end.astype(np.float32)
    X_midi_end = (X_midi_end - X_midi_end.mean(axis=0)) / (X_midi_end.std(axis=0) + 1e-8)
    
    # Combine aux + MIDI for each segment
    # Note: MIDI features expanded from 29 to 64 with advanced features
    X_num_start = np.concatenate([X_aux_start, X_midi_start], axis=1)
    X_num_middle = np.concatenate([X_aux_middle, X_midi_middle], axis=1)
    X_num_end = np.concatenate([X_aux_end, X_midi_end], axis=1)
    
    print(f"Combined numerical features shape: {X_num_start.shape}")
    print(f"  Audio aux: {X_aux_start.shape[1]} features")
    print(f"  MIDI (29 basic + 35 advanced): {X_midi_start.shape[1]} features")
    
    # Train/val/test split
    train_split = config["training"].get("train_split", 0.7)
    val_split = config["training"].get("val_split", 0.15)
    
    n_samples = len(y)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Compute class weights
    class_weights = None
    if config["training"].get("class_weighting", False):
        weight_method = config["training"].get("weight_method", "balanced")
        y_train = y[train_idx]
        
        if weight_method == "inverse_frequency":
            class_weights = compute_class_weights_inverse_frequency(y_train)
        else:
            class_weights = compute_class_weights_balanced(y_train)
        
        print(f"Class weights ({weight_method}): {class_weights}")
    
    # Create datasets
    def create_temporal_dataset(idx):
        inputs = {
            'spec_start': X_spec_start[idx],
            'spec_middle': X_spec_middle[idx],
            'spec_end': X_spec_end[idx],
            'num_feat_start': X_num_start[idx],
            'num_feat_middle': X_num_middle[idx],
            'num_feat_end': X_num_end[idx]
        }
        dataset = tf.data.Dataset.from_tensor_slices((inputs, y[idx]))
        dataset = dataset.shuffle(len(idx))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    train_ds = create_temporal_dataset(train_idx)
    val_ds = create_temporal_dataset(val_idx)
    test_ds = create_temporal_dataset(test_idx)
    
    # Metadata
    metadata = {
        "num_classes": len(composers),
        "class_names": composers,
        "mel_bins": X_spec_start.shape[1],
        "time_frames": X_spec_start.shape[2],
        "num_aux_features": X_aux_start.shape[1],
        "num_midi_features": X_midi_start.shape[1],
        "class_weights": class_weights,
    }
    
    return train_ds, val_ds, test_ds, metadata


def load_temporal_features_raw(config):
    """
    Load raw 3-segment temporal features without splitting into train/val/test.
    Used for k-fold cross-validation where splitting is done per-fold.
    
    Returns raw audio and MIDI arrays combined across all 3 segments.
    
    Args:
        config (dict): Configuration dictionary from load_experiment_config()
    
    Returns:
        tuple: (X_audio, X_midi, y, metadata) where:
            - X_audio: Combined spectrogram features across 3 segments
            - X_midi: Combined MIDI+aux features
            - y: Labels array
            - metadata: Dict with num_classes, class_names, etc.
    """
    
    # Paths
    repo_root = Path(__file__).resolve().parents[2]
    features_dir = resolve_features_dir(repo_root, config)
    labels_path = features_dir / "labels.csv"
    
    # Load labels
    labels_df = pd.read_csv(labels_path, encoding='utf-8')
    print(f"\n{'='*80}")
    print("TEMPORAL FEATURES LOADING")
    print(f"{'='*80}")
    print(f"Total samples in labels.csv: {len(labels_df)}")
    
    # Filter composers if specified
    if "dataset" in config and "composers" in config["dataset"]:
        config_composers = config["dataset"]["composers"]
        print(f"\nComposer filter requested: {config_composers}")
        for comp in config_composers:
            count = len(labels_df[labels_df["composer"] == comp])
            print(f"  - {comp}: {count} samples")
        labels_df = labels_df[labels_df["composer"].isin(config_composers)]
        print(f"\nAfter composer filtering: {len(labels_df)} total samples")
    
    # Create label mapping
    composers = sorted(labels_df["composer"].unique())
    print(f"Composers: {composers}")
    composer_to_idx = {c: i for i, c in enumerate(composers)}
    labels_df["label"] = labels_df["composer"].map(composer_to_idx)
    
    # Load 3-segment features
    spec_start_list, spec_middle_list, spec_end_list = [], [], []
    aux_start_list, aux_middle_list, aux_end_list = [], [], []
    midi_start_list, midi_middle_list, midi_end_list = [], [], []
    labels = []
    missing_count = 0
    loaded_count = 0
    missing_by_composer = {}
    loaded_by_composer = {c: 0 for c in composers}
    
    segments = ["start", "middle", "end"]
    
    print(f"\nLoading feature files...")
    for idx, (_, row) in enumerate(labels_df.iterrows()):
        file_id = str(row["id"]).zfill(4)
        composer = row["composer"]
        
        # Check all required files exist
        required_files = []
        for seg in segments:
            required_files.append(features_dir / "audio" / f"{file_id}_mel_{seg}.npy")
            required_files.append(features_dir / "audio" / f"{file_id}_aux_{seg}.npy")
            required_files.append(features_dir / "midi" / f"{file_id}_midi_{seg}.npy")
        
        if not all(f.exists() for f in required_files):
            missing_count += 1
            if composer not in missing_by_composer:
                missing_by_composer[composer] = 0
            missing_by_composer[composer] += 1
            if missing_count <= 3:
                print(f"  Warning: Missing files for {file_id} ({composer})")
            continue
        
        try:
            # Load spectrograms
            spec_start = np.load(features_dir / "audio" / f"{file_id}_mel_start.npy")
            spec_middle = np.load(features_dir / "audio" / f"{file_id}_mel_middle.npy")
            spec_end = np.load(features_dir / "audio" / f"{file_id}_mel_end.npy")
            
            # Load auxiliary features
            aux_start = np.load(features_dir / "audio" / f"{file_id}_aux_start.npy")
            aux_middle = np.load(features_dir / "audio" / f"{file_id}_aux_middle.npy")
            aux_end = np.load(features_dir / "audio" / f"{file_id}_aux_end.npy")
            
            # Load MIDI features
            midi_start = np.load(features_dir / "midi" / f"{file_id}_midi_start.npy")
            midi_middle = np.load(features_dir / "midi" / f"{file_id}_midi_middle.npy")
            midi_end = np.load(features_dir / "midi" / f"{file_id}_midi_end.npy")
            
            # Append to lists
            spec_start_list.append(spec_start)
            spec_middle_list.append(spec_middle)
            spec_end_list.append(spec_end)
            
            aux_start_list.append(aux_start)
            aux_middle_list.append(aux_middle)
            aux_end_list.append(aux_end)
            
            midi_start_list.append(midi_start)
            midi_middle_list.append(midi_middle)
            midi_end_list.append(midi_end)
            
            labels.append(row["label"])
            loaded_count += 1
            loaded_by_composer[composer] += 1
            
        except Exception as e:
            print(f"Error loading {file_id}: {e}")
            continue
    
    if len(labels) == 0:
        raise ValueError("No valid temporal features found!")
    
    # Print loading summary
    print(f"\nFeature loading complete:")
    print(f"  Successfully loaded: {loaded_count}")
    print(f"  Missing files: {missing_count}")
    print(f"  Failed to load: {len(labels_df) - loaded_count - missing_count}")
    
    if loaded_by_composer:
        print(f"\nLoaded samples per composer:")
        for composer in sorted(loaded_by_composer.keys()):
            loaded = loaded_by_composer[composer]
            missing = missing_by_composer.get(composer, 0)
            print(f"  - {composer}: {loaded} loaded, {missing} missing")
    
    # Convert to arrays
    X_spec_start = np.array(spec_start_list)
    X_spec_middle = np.array(spec_middle_list)
    X_spec_end = np.array(spec_end_list)
    
    X_aux_start = np.array(aux_start_list)
    X_aux_middle = np.array(aux_middle_list)
    X_aux_end = np.array(aux_end_list)
    
    X_midi_start = np.array(midi_start_list)
    X_midi_middle = np.array(midi_middle_list)
    X_midi_end = np.array(midi_end_list)
    
    y = np.array(labels, dtype=np.int32)
    
    print(f"\n=== Raw Temporal Data Shapes ===")
    print(f"Spectrograms: {X_spec_start.shape}")
    print(f"Aux features: {X_aux_start.shape}")
    print(f"MIDI features: {X_midi_start.shape}")
    print(f"Labels: {y.shape}, unique: {np.unique(y)}")
    
    # Add channel dimension to spectrograms if needed
    if len(X_spec_start.shape) == 3:
        X_spec_start = X_spec_start[..., np.newaxis]
        X_spec_middle = X_spec_middle[..., np.newaxis]
        X_spec_end = X_spec_end[..., np.newaxis]
    
    # Convert to float32
    X_spec_start = X_spec_start.astype(np.float32)
    X_spec_middle = X_spec_middle.astype(np.float32)
    X_spec_end = X_spec_end.astype(np.float32)
    
    X_aux_start = X_aux_start.astype(np.float32)
    X_aux_middle = X_aux_middle.astype(np.float32)
    X_aux_end = X_aux_end.astype(np.float32)
    
    X_midi_start = X_midi_start.astype(np.float32)
    X_midi_middle = X_midi_middle.astype(np.float32)
    X_midi_end = X_midi_end.astype(np.float32)
    
    # Combine aux + MIDI for each segment
    X_num_start = np.concatenate([X_aux_start, X_midi_start], axis=1)
    X_num_middle = np.concatenate([X_aux_middle, X_midi_middle], axis=1)
    X_num_end = np.concatenate([X_aux_end, X_midi_end], axis=1)
    
    print(f"Numerical features shape per segment: {X_num_start.shape}")
    
    # Store all segments separately (will be normalized per-fold)
    temporal_data = {
        'spec_start': X_spec_start,
        'spec_middle': X_spec_middle,
        'spec_end': X_spec_end,
        'num_start': X_num_start,
        'num_middle': X_num_middle,
        'num_end': X_num_end
    }
    
    # Metadata
    metadata = {
        "num_classes": len(composers),
        "class_names": composers,
        "mel_bins": X_spec_start.shape[1],
        "time_frames": X_spec_start.shape[2],
        "num_aux_features": X_aux_start.shape[1],
        "num_midi_features": X_midi_start.shape[1],
    }
    
    return temporal_data, y, metadata


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
    features_dir = resolve_features_dir(repo_root, config)
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
    y = np.array(labels, dtype=np.int32)                # Shape: (N,)
    
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
    
    # Class weights (optional)
    class_weights = None
    if config["training"].get("class_weighting", False):
        # Choose weight computation method
        weight_method = config["training"].get("weight_method", "balanced")
        y_train = y[train_idx]
        
        if weight_method == "inverse_frequency":
            class_weights = compute_class_weights_inverse_frequency(y_train)
        else:  # default: "balanced"
            class_weights = compute_class_weights_balanced(y_train)
        
        print(f"Class weight method: {weight_method}")
        print(f"Computed class weights: {class_weights}")

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
        "class_weights": class_weights,
    }
    
    return train_ds, val_ds, test_ds, metadata