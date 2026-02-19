"""
train_temporal.py
-------------------------
Training script for temporal multimodal composer classification model (3-segment architecture).

To run from command line:
python -m src.training.train_temporal configs/[config].yaml
"""

import os
import json
import yaml
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Enable mixed precision for GPU
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"✓ Mixed precision enabled on {len(gpus)} GPU(s)")
    else:
        print("CPU-only training - using float32")
except Exception as e:
    print(f"Note: Mixed precision not enabled - {e}")

# Imports
if __name__ == "__main__" and __package__ is None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from configs.load_config import load_experiment_config
    from models.temporal_multimodal import build_temporal_multimodal_model, build_temporal_multimodal_model_simple
    from training.load_features import get_temporal_datasets
    from utils.losses import (
        weighted_sparse_categorical_crossentropy,
        focal_loss,
        compute_class_weights_balanced,
        compute_class_weights_inverse_frequency
    )
else:
    from ..configs.load_config import load_experiment_config
    from ..models.temporal_multimodal import build_temporal_multimodal_model, build_temporal_multimodal_model_simple
    from .load_features import get_temporal_datasets
    from ..utils.losses import (
        weighted_sparse_categorical_crossentropy,
        focal_loss,
        compute_class_weights_balanced,
        compute_class_weights_inverse_frequency
    )


def build_model_from_config(config, metadata):
    """Build temporal multimodal model from configuration."""
    
    # Get architecture parameters
    arch_type = config.get("model", {}).get("architecture", "full")
    
    num_classes = metadata["num_classes"]
    mel_bins = metadata["mel_bins"]
    time_frames = metadata["time_frames"]
    num_aux_features = metadata["num_aux_features"]
    num_midi_features = metadata["num_midi_features"]
    
    # Model hyperparameters
    model_params = config.get("model", {})
    cnn_embedding_dim = model_params.get("cnn_embedding_dim", 256)
    numerical_embedding_dim = model_params.get("numerical_embedding_dim", 128)
    lstm_units = model_params.get("lstm_units", 128)
    dropout_rate = model_params.get("dropout_rate", 0.4)
    use_shared_cnn = model_params.get("use_shared_cnn", True)
    
    # Build model
    if arch_type == "simple":
        print("Building simplified temporal model...")
        model = build_temporal_multimodal_model_simple(
            mel_bins=mel_bins,
            time_frames=time_frames,
            num_aux_features=num_aux_features,
            num_midi_features=num_midi_features,
            num_classes=num_classes,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate
        )
    else:
        print("Building full temporal multimodal model...")
        model = build_temporal_multimodal_model(
            mel_bins=mel_bins,
            time_frames=time_frames,
            num_aux_features=num_aux_features,
            num_midi_features=num_midi_features,
            num_classes=num_classes,
            cnn_embedding_dim=cnn_embedding_dim,
            numerical_embedding_dim=numerical_embedding_dim,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            use_shared_cnn=use_shared_cnn
        )
    
    return model


def compile_model(model, config, class_weights=None):
    """Compile model with loss function and optimizer."""
    
    # Get loss configuration
    loss_config = config.get("loss", {})
    loss_type = loss_config.get("type", "sparse_categorical_crossentropy")
    
    # Build loss function
    if loss_type == "weighted":
        if class_weights is None:
            print("Warning: Weighted loss requested but no class weights provided")
            loss_fn = "sparse_categorical_crossentropy"
        else:
            loss_fn = weighted_sparse_categorical_crossentropy(class_weights)
            print(f"Using weighted crossentropy with class weights")
    
    elif loss_type == "focal":
        alpha = loss_config.get("alpha", 0.25)
        gamma = loss_config.get("gamma", 2.0)
        loss_fn = focal_loss(alpha=alpha, gamma=gamma)
        print(f"Using focal loss (alpha={alpha}, gamma={gamma})")
    
    else:
        loss_fn = "sparse_categorical_crossentropy"
        print("Using standard sparse categorical crossentropy")
    
    # Optimizer
    learning_rate = config["training"].get("learning_rate", 0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )
    
    return model


def train_single_fold(model, train_ds, val_ds, config, fold_num=None):
    """Train model for one fold."""
    
    epochs = config["training"].get("epochs", 50)
    patience = config["training"].get("patience", 10)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, test_ds, class_names):
    """Evaluate model and return metrics."""
    
    # Get predictions
    y_true = []
    y_pred = []
    
    for inputs, labels in test_ds:
        predictions = model.predict(inputs, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    
    print(f"\n=== Test Metrics ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    
    return metrics


def save_model_and_results(model, metrics, config, metadata, fold_num=None):
    """Save trained model and results."""
    
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "outputs"
    models_dir = output_dir / "models"
    results_dir = output_dir / "results"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    trial_name = config.get("trial_name", "temporal_trial")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if fold_num is not None:
        base_name = f"{trial_name}_fold{fold_num}_{timestamp}"
    else:
        base_name = f"{trial_name}_{timestamp}"
    
    # Save model
    model_path = models_dir / f"{base_name}.keras"
    model.save(model_path)
    print(f"\n✓ Model saved: {model_path}")
    
    # Save results
    results = {
        "trial_name": trial_name,
        "timestamp": timestamp,
        "fold": fold_num,
        "config": config,
        "metadata": metadata,
        "metrics": metrics
    }
    
    results_path = results_dir / f"{base_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved: {results_path}")
    
    return model_path, results_path


def main(config_path):
    """Main training function."""
    
    print("="*80)
    print("TEMPORAL MULTIMODAL TRAINING")
    print("="*80)
    
    # Load config
    config = load_experiment_config(config_path)
    print(f"\n✓ Loaded config: {config_path}")
    print(f"  Trial name: {config.get('trial_name', 'unnamed')}")
    
    # Load datasets
    print("\n" + "="*80)
    print("LOADING TEMPORAL DATASETS")
    print("="*80)
    
    train_ds, val_ds, test_ds, metadata = get_temporal_datasets(config)
    
    print(f"\n✓ Datasets loaded")
    print(f"  Classes: {metadata['num_classes']}")
    print(f"  Mel bins: {metadata['mel_bins']}")
    print(f"  Time frames: {metadata['time_frames']}")
    print(f"  Aux features: {metadata['num_aux_features']}")
    print(f"  MIDI features: {metadata['num_midi_features']}")
    
    # Build model
    print("\n" + "="*80)
    print("BUILDING MODEL")
    print("="*80)
    
    model = build_model_from_config(config, metadata)
    model = compile_model(model, config, class_weights=metadata.get("class_weights"))
    
    print(f"\n✓ Model built and compiled")
    print(f"  Parameters: {model.count_params():,}")
    
    # Print model summary
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    model.summary()
    
    # Train
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    history = train_single_fold(model, train_ds, val_ds, config)
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    metrics = evaluate_model(model, test_ds, metadata["class_names"])
    
    # Save
    print("\n" + "="*80)
    print("SAVING")
    print("="*80)
    
    model_path, results_path = save_model_and_results(model, metrics, config, metadata)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"✓ Model: {model_path}")
    print(f"✓ Results: {results_path}")
    print(f"✓ Test Accuracy: {metrics['accuracy']:.4f}")
    
    return model, metrics


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.training.train_temporal <config_path>")
        print("Example: python -m src.training.train_temporal configs/temporal_trial1.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path)
