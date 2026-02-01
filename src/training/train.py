"""
train.py
-------------------------
Training script for composer classification models.

To run from command line:
python -m src.training.train configs/[trial].yaml
"""

import os
import json
import yaml
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path

# Enable mixed precision training for GPU acceleration (only on GPU)
# CPU + float16 is VERY slow without AVX-512 support
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Only enable mixed precision if GPU is available
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"✓ Mixed precision training enabled (float16) on {len(gpus)} GPU(s)")
    else:
        print("CPU-only training detected - using float32 (faster without AVX-512 GPU support)")
except Exception as e:
    print(f"Note: Could not enable mixed precision - {e}")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold

# Handle imports whether run as module or directly
if __name__ == "__main__" and __package__ is None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from configs.load_config import load_experiment_config
    from models.multimodal import build_multimodal_model
    from training.load_features import get_datasets
    from visualization.confusion_matrix import plot_all_confusion_matrices
else:
    from ..configs.load_config import load_experiment_config
    from ..models.multimodal import build_multimodal_model
    from .load_features import get_datasets
    from ..visualization.confusion_matrix import plot_all_confusion_matrices

# ===== Helper Functions =====

def load_all_features(config):
    """Load all features without train/val/test split for k-fold CV."""
    import pandas as pd
    
    # Paths
    repo_root = Path(__file__).resolve().parents[2]
    features_dir = repo_root / "results" / "features"
    labels_path = features_dir / "labels.csv"
    
    # Load labels
    labels_df = pd.read_csv(labels_path, encoding='utf-8')
    
    # Filter composers based on config
    if "dataset" in config and "composers" in config["dataset"]:
        config_composers = config["dataset"]["composers"]
        labels_df = labels_df[labels_df["composer"].isin(config_composers)]
    
    # Get unique composers and create label mapping
    composers = sorted(labels_df["composer"].unique())
    composer_to_idx = {c: i for i, c in enumerate(composers)}
    labels_df["label"] = labels_df["composer"].map(composer_to_idx)
    
    # Load features
    audio_features = []
    midi_features = []
    labels = []
    
    for _, row in labels_df.iterrows():
        file_id = str(row["id"]).zfill(4)
        mel_path = features_dir / "audio" / f"{file_id}_mel.npy"
        midi_path = features_dir / "midi" / f"{file_id}_midi.npy"
        
        if not mel_path.exists() or not midi_path.exists():
            continue
        
        try:
            mel = np.load(mel_path)
            midi = np.load(midi_path)
            audio_features.append(mel)
            midi_features.append(midi)
            labels.append(row["label"])
        except Exception:
            continue
    
    # Convert to arrays
    X_audio = np.array(audio_features)
    X_midi = np.array(midi_features)
    y = np.array(labels)
    
    # Ensure 4D shape for audio
    if len(X_audio.shape) == 3:
        X_audio = X_audio[..., np.newaxis]
    
    # Normalize
    X_audio = X_audio.astype(np.float32)
    X_audio = (X_audio - X_audio.mean()) / (X_audio.std() + 1e-8)
    
    X_midi = X_midi.astype(np.float32)
    if len(X_midi.shape) == 1:
        X_midi = X_midi.reshape(-1, 1)
    X_midi = (X_midi - X_midi.mean(axis=0)) / (X_midi.std(axis=0) + 1e-8)
    
    metadata = {
        "num_classes": len(composers),
        "class_names": composers,
        "feature_dim": X_midi.shape[1],
        "audio_shape": tuple(X_audio.shape[1:]),
    }
    
    return X_audio, X_midi, y, metadata


# ===== Load Config & Setup Directories =====

def setup_run_dirs(config):
    trial = config["experiment"]["trial"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = f"{trial}_{timestamp}"
    base_out = config["output"]["results_dir"]

    run_dir = os.path.join(base_out, run_name)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


# ===== Build Model =====

def build_model(config, num_classes, feature_dim, audio_shape):
    """Build multimodal model with audio and MIDI features."""
    # Extract audio shape components
    mel_bins = audio_shape[0]
    time_frames = audio_shape[1]
    
    model = build_multimodal_model(
        mel_bins=mel_bins,
        time_frames=time_frames,
        num_engineered_features=feature_dim,
        num_classes=num_classes
    )

    return model


# ===== Compile Model =====

def compile_model(model, config):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["training"]["learning_rate"]
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


# ===== Train Model =====

def train_model(model, train_ds, val_ds, config):
    callbacks = []

    # Early stopping configuration
    early_stop_config = config["training"].get("early_stopping", {})
    if early_stop_config.get("enabled", False):
        patience = early_stop_config.get("patience", 5)
        min_delta = early_stop_config.get("min_delta", 0.001)
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=1
            )
        )
        print(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["epochs"],
        callbacks=callbacks,
        verbose=1
    )

    return history


# ===== Evaluate Model & Save Metrics =====

def evaluate_model(model, test_ds, run_dir, class_names):
    y_true = []
    y_pred = []

    for batch in test_ds:
        (audio, features), labels = batch
        preds = model.predict([audio, features], verbose=0)
        preds = np.argmax(preds, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

    # Save metrics
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    np.save(os.path.join(run_dir, "confusion_matrix.npy"), cm)

    # Also save labeled version
    cm_labeled = {
        "labels": class_names,
        "matrix": cm.tolist()
    }
    with open(os.path.join(run_dir, "confusion_matrix.json"), "w") as f:
        json.dump(cm_labeled, f, indent=4)

    return metrics


# ===== Save Model =====

def save_model(model, run_dir, config):
    """Save the trained model to outputs/models directory."""
    trial = config["experiment"]["trial"]
    timestamp = os.path.basename(run_dir).split('_', 1)[-1]  # Extract timestamp from run_dir
    model_name = f"{trial}_{timestamp}.keras"
    
    # Create models directory
    models_dir = Path("outputs/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = models_dir / model_name
    model.save(str(model_path))
    
    print(f"Model saved to: {model_path}")
    
    # Also save model path to run directory for reference
    model_info = {
        "model_path": str(model_path),
        "model_name": model_name,
        "trial": trial
    }
    with open(os.path.join(run_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=4)
    
    return str(model_path)


# ===== K-Fold Cross Validation =====

def train_with_kfold(config, run_dir):
    """Train model using k-fold cross validation."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    n_folds = config["training"].get("k_folds", 5)
    batch_size = config["training"].get("batch_size", 8)
    
    print(f"\n{'='*60}")
    print(f"Starting {n_folds}-Fold Cross Validation")
    print(f"{'='*60}\n")
    
    # Load all features
    X_audio, X_midi, y, metadata = load_all_features(config)
    
    num_classes = metadata["num_classes"]
    class_names = metadata["class_names"]
    feature_dim = metadata["feature_dim"]
    audio_shape = metadata["audio_shape"]
    
    print(f"Total samples: {len(y)}")
    print(f"Classes: {class_names}")
    print(f"Samples per class: {np.bincount(y)}\n")
    
    # K-Fold split (stratified to maintain class balance)
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_metrics = []
    fold_cms = []
    
    # Collect all predictions across folds for combined confusion matrix
    all_y_true = []
    all_y_pred = []
    
    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(X_audio, y), 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{n_folds}")
        print(f"{'='*60}")
        
        # Further split train_val into train and validation
        n_train = int(len(train_val_idx) * 0.85)  # 85% train, 15% val
        rng = np.random.RandomState(42 + fold)  # Unique seed per fold
        rng.shuffle(train_val_idx)
        train_idx = train_val_idx[:n_train]
        val_idx = train_val_idx[n_train:]
        
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # Create datasets for this fold
        def create_dataset(x_audio, x_midi, y_labels):
            dataset = tf.data.Dataset.from_tensor_slices(((x_audio, x_midi), y_labels))
            dataset = dataset.shuffle(len(x_audio))
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            return dataset
        
        train_ds = create_dataset(X_audio[train_idx], X_midi[train_idx], y[train_idx])
        val_ds = create_dataset(X_audio[val_idx], X_midi[val_idx], y[val_idx])
        test_ds = create_dataset(X_audio[test_idx], X_midi[test_idx], y[test_idx])
        
        # Build and compile model
        model = build_model(config, num_classes, feature_dim, audio_shape)
        compile_model(model, config)
        
        # Train
        print(f"\nTraining fold {fold}...")
        train_model(model, train_ds, val_ds, config)
        
        # Evaluate on test set
        y_true = []
        y_pred = []
        
        for batch in test_ds:
            (audio, features), labels = batch
            preds = model.predict([audio, features], verbose=0)
            preds = np.argmax(preds, axis=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Add to combined predictions
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
        # Calculate metrics
        metrics = {
            "fold": fold,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
        
        fold_metrics.append(metrics)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fold_cms.append(cm)
        
        # Print fold results
        print(f"\nFold {fold} Results:")
        for k, v in metrics.items():
            if k != "fold":
                print(f"  {k}: {v:.4f}")
        
        # Save fold-specific results
        fold_dir = os.path.join(run_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        
        np.save(os.path.join(fold_dir, "confusion_matrix.npy"), cm)
        
        cm_labeled = {
            "labels": class_names,
            "matrix": cm.tolist()
        }
        with open(os.path.join(fold_dir, "confusion_matrix.json"), "w") as f:
            json.dump(cm_labeled, f, indent=4)
        
        # Save model for this fold
        model_path = Path("outputs/models") / f"{config['experiment']['trial']}_fold{fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        print(f"  Model saved to: {model_path}")
    
    # Aggregate results across folds
    print(f"\n{'='*60}")
    print("Cross-Validation Results (Mean ± Std)")
    print(f"{'='*60}")
    
    aggregated_metrics = {}
    metric_keys = [k for k in fold_metrics[0].keys() if k != "fold"]
    
    for key in metric_keys:
        values = [m[key] for m in fold_metrics]
        aggregated_metrics[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "values": values
        }
        print(f"{key}: {aggregated_metrics[key]['mean']:.4f} ± {aggregated_metrics[key]['std']:.4f}")
    
    # Save aggregated results
    with open(os.path.join(run_dir, "cv_results.json"), "w") as f:
        json.dump(aggregated_metrics, f, indent=4)
    
    with open(os.path.join(run_dir, "fold_metrics.json"), "w") as f:
        json.dump(fold_metrics, f, indent=4)
    
    # Create combined confusion matrix from all fold predictions
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    combined_cm = confusion_matrix(all_y_true, all_y_pred)
    
    print(f"\n{'='*60}")
    print("Combined Confusion Matrix (All Folds)")
    print(f"{'='*60}")
    print(f"Total predictions: {len(all_y_true)}")
    print(f"Overall accuracy: {accuracy_score(all_y_true, all_y_pred):.4f}\n")
    
    # Save combined confusion matrix (main result)
    np.save(os.path.join(run_dir, "confusion_matrix.npy"), combined_cm)
    
    combined_cm_labeled = {
        "labels": class_names,
        "matrix": combined_cm.tolist(),
        "description": "Combined confusion matrix aggregated from all fold predictions"
    }
    with open(os.path.join(run_dir, "confusion_matrix.json"), "w") as f:
        json.dump(combined_cm_labeled, f, indent=4)
    
    # Also save average confusion matrix for reference
    avg_cm = np.mean(fold_cms, axis=0)
    np.save(os.path.join(run_dir, "avg_confusion_matrix.npy"), avg_cm)
    
    avg_cm_labeled = {
        "labels": class_names,
        "matrix": avg_cm.tolist(),
        "description": "Average of individual fold confusion matrices"
    }
    with open(os.path.join(run_dir, "avg_confusion_matrix.json"), "w") as f:
        json.dump(avg_cm_labeled, f, indent=4)
    
    print(f"\nResults saved to: {run_dir}")
    return aggregated_metrics


# ===== Main Training Pipeline =====

def main(trial_config_path):
    # Load config
    config = load_experiment_config(trial_config_path)

    run_dir = setup_run_dirs(config)

    # Save config snapshot
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Check if simple split is explicitly requested (otherwise use k-fold by default)
    use_simple_split = config["training"].get("use_simple_split", False)
    k_folds = config["training"].get("k_folds", 5)
    
    if use_simple_split or k_folds is None:
        # Standard train/val/test split (only if explicitly requested)
        # Load datasets
        train_ds, val_ds, test_ds, metadata = get_datasets(config)

        num_classes = metadata["num_classes"]
        class_names = metadata["class_names"]
        feature_dim = metadata["feature_dim"]
        audio_shape = metadata["audio_shape"]

        # Build + compile
        model = build_model(config, num_classes, feature_dim, audio_shape)
        compile_model(model, config)

        # Train
        train_model(model, train_ds, val_ds, config)

        # Evaluate
        metrics = evaluate_model(model, test_ds, run_dir, class_names)

        print("Final metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        # Save model
        model_path = save_model(model, run_dir, config)

        # Automatically generate confusion matrix plots for this run
        plot_all_confusion_matrices(run_dir)
    else:
        # K-fold cross validation (default)
        train_with_kfold(config, run_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/base.yaml"
    main(config_path)
