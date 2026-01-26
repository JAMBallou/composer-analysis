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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

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

    if config["training"].get("early_stopping", {}).get("enabled", False):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                patience=config["training"]["early_stopping"]["patience"],
                restore_best_weights=True
            )
        )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["epochs"],
        callbacks=callbacks
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


# ===== Main Training Pipeline =====

def main(trial_config_path):
    # Load config
    config = load_experiment_config(trial_config_path)

    run_dir = setup_run_dirs(config)

    # Save config snapshot
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

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

    # Automatically generate confusion matrix plots
    plot_all_confusion_matrices(
        results_dir=config["output"]["results_dir"],
        output_dir=os.path.join(
            config["output"]["results_dir"],
            "confusion_matrices"
        )
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/base.yaml"
    main(config_path)
