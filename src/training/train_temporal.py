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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# Configure GPU for optimal performance
try:
    from ..utils.gpu_config import auto_configure_gpu
    gpu_config = auto_configure_gpu(verbose=True)
except Exception as e:
    print(f"⚠️  GPU configuration error: {e}")
    print("   Proceeding with default TensorFlow settings...")
    gpu_config = {'gpus_configured': []}

# Imports
if __name__ == "__main__" and __package__ is None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from configs.load_config import load_experiment_config
    from models.temporal_multimodal import build_temporal_multimodal_model, build_temporal_multimodal_model_simple
    from training.load_features import get_temporal_datasets, load_temporal_features_raw
    from utils.losses import (
        weighted_sparse_categorical_crossentropy,
        focal_loss,
        compute_class_weights_balanced,
        compute_class_weights_inverse_frequency
    )
else:
    from ..configs.load_config import load_experiment_config
    from ..models.temporal_multimodal import build_temporal_multimodal_model, build_temporal_multimodal_model_simple
    from .load_features import get_temporal_datasets, load_temporal_features_raw
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


def get_trial_id(trial_name: str) -> str:
    if trial_name.startswith("temporal_"):
        trial_name = trial_name.replace("temporal_", "", 1)
    return trial_name.split("_", 1)[0]


def save_fold_results(model, metrics, config, metadata, run_dir, fold_num, class_names):
    """Save fold-specific model, metrics, and confusion matrix."""
    
    # Extract run name from run_dir (last component of path)
    run_name = Path(run_dir).name
    
    # Create models folder organized by run in outputs/models/
    repo_root = Path(__file__).resolve().parents[2]
    models_base = repo_root / "outputs" / "models" / run_name
    models_base.mkdir(parents=True, exist_ok=True)
    
    fold_dir = Path(run_dir) / f"fold_{fold_num}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    trial_name = config.get("trial_name", "temporal_trial")
    trial_id = get_trial_id(trial_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{trial_id}_fold{fold_num}_{timestamp}"
    
    # Save model
    model_path = models_base / f"{base_name}.keras"
    model.save(model_path)
    print(f"  + Model saved: {model_path}")
    
    # Save fold metrics
    metrics_to_save = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    
    # Save confusion matrix as NPY
    cm = np.array(metrics["confusion_matrix"])
    np.save(fold_dir / "confusion_matrix.npy", cm)
    
    # Save confusion matrix as JSON with labels
    cm_labeled = {
        "labels": class_names,
        "matrix": cm.tolist()
    }
    with open(fold_dir / "confusion_matrix.json", "w") as f:
        json.dump(cm_labeled, f, indent=2)
    
    return model_path


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
    trial_id = get_trial_id(trial_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if fold_num is not None:
        base_name = f"{trial_id}_fold{fold_num}_{timestamp}"
    else:
        base_name = f"{trial_id}_{timestamp}"
    
    # Save model
    model_path = models_dir / f"{base_name}.keras"
    model.save(model_path)
    print(f"\n+ Model saved: {model_path}")
    
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
    
    print(f"+ Results saved: {results_path}")
    
    return model_path, results_path


def main(config_path):
    """Main training function with optional k-fold cross-validation."""
    
    print("="*80)
    print("TEMPORAL MULTIMODAL TRAINING")
    print("="*80)
    
    # Load config
    config = load_experiment_config(config_path)
    print(f"\n+ Loaded config: {config_path}")
    print(f"  Trial name: {config.get('trial_name', 'unnamed')}")
    
    # Check if k-fold CV is enabled
    k_folds = config.get("training", {}).get("k_folds", 1)
    use_kfold = k_folds > 1
    
    if use_kfold:
        print(f"\n+ K-Fold Cross-Validation enabled: {k_folds} folds")
        return main_kfold(config_path, k_folds)
    else:
        print(f"\n+ Single fold training (no cross-validation)")
        return main_single_fold(config_path)


def main_single_fold(config_path):
    """Train a single fold without cross-validation."""
    
    # Load config
    config = load_experiment_config(config_path)
    
    # Load datasets
    print("\n" + "="*80)
    print("LOADING TEMPORAL DATASETS")
    print("="*80)
    
    train_ds, val_ds, test_ds, metadata = get_temporal_datasets(config)
    
    print(f"\n+ Datasets loaded")
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
    
    print(f"\n+ Model built and compiled")
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
    print(f"+ Model: {model_path}")
    print(f"+ Results: {results_path}")
    print(f"+ Test Accuracy: {metrics['accuracy']:.4f}")
    
    return model, metrics


def main_kfold(config_path, k_folds):
    """Train with k-fold cross-validation."""
    
    # Load config
    config = load_experiment_config(config_path)
    
    # Setup output directory for this run
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "outputs"
    trial_name = config.get("trial_name", "temporal_trial")
    trial_id = get_trial_id(trial_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{trial_id}_{timestamp}"
    run_dir = output_dir / "results" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n+ K-Fold CV run directory: {run_dir}")
    
    # Load full dataset (without splitting into train/val/test)
    print("\n" + "="*80)
    print("LOADING TEMPORAL DATASETS FOR K-FOLD CV")
    print("="*80)
    
    # Load raw data for stratified k-fold split
    temporal_data, y, metadata = load_temporal_features_raw(config)
    
    if len(y) == 0:
        raise ValueError(
            "No temporal samples loaded. Verify feature extraction output and composer filters."
        )

    print(f"\n+ Datasets loaded")
    print(f"  Total samples: {len(y)}")
    print(f"  Classes: {metadata['num_classes']}")
    print(f"  Samples per class: {np.bincount(y, minlength=metadata['num_classes'])}")
    print(f"  Spectrogram shape per segment: {temporal_data['spec_start'].shape}")
    print(f"  Features shape per segment: {temporal_data['num_start'].shape}")
    
    class_names = metadata["class_names"]
    num_classes = metadata["num_classes"]

    # Initialize k-fold
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_metrics_list = []
    fold_cms = []
    all_y_true = []
    all_y_pred = []
    
    # Iterate over folds
    for fold_num, (train_val_idx, test_idx) in enumerate(kfold.split(temporal_data['spec_start'], y), 1):
        
        print(f"\n{'='*80}")
        print(f"FOLD {fold_num}/{k_folds}")
        print(f"{'='*80}")
        
        # Further split train_val into train and validation
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42 + fold_num)
        train_rel_idx, val_rel_idx = next(sss.split(train_val_idx, y[train_val_idx]))
        train_idx = train_val_idx[train_rel_idx]
        val_idx = train_val_idx[val_rel_idx]
        
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # Per-fold normalization (fit on train only to avoid leakage)
        # Normalize each segment separately
        def normalize_segment(data, indices_train):
            mean = data[indices_train].mean()
            std = data[indices_train].std() + 1e-8
            return mean, std
        
        spec_start_mean, spec_start_std = normalize_segment(temporal_data['spec_start'], train_idx)
        spec_middle_mean, spec_middle_std = normalize_segment(temporal_data['spec_middle'], train_idx)
        spec_end_mean, spec_end_std = normalize_segment(temporal_data['spec_end'], train_idx)
        
        num_start_mean = temporal_data['num_start'][train_idx].mean(axis=0)
        num_start_std = temporal_data['num_start'][train_idx].std(axis=0) + 1e-8
        num_middle_mean = temporal_data['num_middle'][train_idx].mean(axis=0)
        num_middle_std = temporal_data['num_middle'][train_idx].std(axis=0) + 1e-8
        num_end_mean = temporal_data['num_end'][train_idx].mean(axis=0)
        num_end_std = temporal_data['num_end'][train_idx].std(axis=0) + 1e-8
        
        # Normalize all splits
        def normalize_fold_data(idx):
            return {
                'spec_start': (temporal_data['spec_start'][idx] - spec_start_mean) / spec_start_std,
                'spec_middle': (temporal_data['spec_middle'][idx] - spec_middle_mean) / spec_middle_std,
                'spec_end': (temporal_data['spec_end'][idx] - spec_end_mean) / spec_end_std,
                'num_feat_start': (temporal_data['num_start'][idx] - num_start_mean) / num_start_std,
                'num_feat_middle': (temporal_data['num_middle'][idx] - num_middle_mean) / num_middle_std,
                'num_feat_end': (temporal_data['num_end'][idx] - num_end_mean) / num_end_std
            }
        
        train_data = normalize_fold_data(train_idx)
        val_data = normalize_fold_data(val_idx)
        test_data = normalize_fold_data(test_idx)
        
        # Create datasets for this fold
        def create_temporal_dataset(data_dict, labels):
            batch_size = config["training"]["batch_size"]
            dataset = tf.data.Dataset.from_tensor_slices((data_dict, labels))
            dataset = dataset.shuffle(len(labels))
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            return dataset
        
        train_ds = create_temporal_dataset(train_data, y[train_idx])
        val_ds = create_temporal_dataset(val_data, y[val_idx])
        test_ds = create_temporal_dataset(test_data, y[test_idx])
        
        # Build and compile model
        print(f"\nBuilding model for fold {fold_num}...")
        model = build_model_from_config(config, metadata)
        
        # Compute class weights for this fold
        class_weights_fold = None
        if config["training"].get("class_weighting", False):
            y_train = y[train_idx]
            weight_method = config["training"].get("weight_method", "balanced")
            if weight_method == "inverse_frequency":
                class_weights_fold = compute_class_weights_inverse_frequency(y_train)
            else:
                class_weights_fold = compute_class_weights_balanced(y_train)
            print(f"Class weights: {class_weights_fold}")
        
        model = compile_model(model, config, class_weights=class_weights_fold)
        
        # Train
        print(f"\nTraining fold {fold_num}...")
        train_single_fold(model, train_ds, val_ds, config)
        
        # Evaluate on test set
        print(f"\nEvaluating fold {fold_num}...")
        y_true_fold = []
        y_pred_fold = []
        
        for batch in test_ds:
            inputs, labels = batch
            preds = model.predict(inputs, verbose=0)
            preds = np.argmax(preds, axis=1)
            y_true_fold.extend(labels.numpy())
            y_pred_fold.extend(preds)
        
        y_true_fold = np.array(y_true_fold)
        y_pred_fold = np.array(y_pred_fold)
        
        # Add to combined predictions
        all_y_true.extend(y_true_fold)
        all_y_pred.extend(y_pred_fold)
        
        # Calculate metrics
        fold_metrics = {
            "accuracy": float(accuracy_score(y_true_fold, y_pred_fold)),
            "precision": float(precision_score(y_true_fold, y_pred_fold, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true_fold, y_pred_fold, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true_fold, y_pred_fold, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true_fold, y_pred_fold).tolist()
        }
        
        fold_metrics_list.append(fold_metrics)
        fold_cms.append(np.array(fold_metrics["confusion_matrix"]))
        
        print(f"\nFold {fold_num} Results:")
        print(f"  Accuracy:  {fold_metrics['accuracy']:.4f}")
        print(f"  Precision: {fold_metrics['precision']:.4f}")
        print(f"  Recall:    {fold_metrics['recall']:.4f}")
        print(f"  F1:        {fold_metrics['f1']:.4f}")
        
        # Save fold-specific results
        save_fold_results(model, fold_metrics, config, metadata, run_dir, fold_num, class_names)
    
    # Aggregate results across folds
    print(f"\n{'='*80}")
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*80}")
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    aggregated_metrics = {
        "accuracy": {
            "mean": float(np.mean([m["accuracy"] for m in fold_metrics_list])),
            "std": float(np.std([m["accuracy"] for m in fold_metrics_list])),
            "values": [m["accuracy"] for m in fold_metrics_list]
        },
        "precision": {
            "mean": float(np.mean([m["precision"] for m in fold_metrics_list])),
            "std": float(np.std([m["precision"] for m in fold_metrics_list])),
            "values": [m["precision"] for m in fold_metrics_list]
        },
        "recall": {
            "mean": float(np.mean([m["recall"] for m in fold_metrics_list])),
            "std": float(np.std([m["recall"] for m in fold_metrics_list])),
            "values": [m["recall"] for m in fold_metrics_list]
        },
        "f1": {
            "mean": float(np.mean([m["f1"] for m in fold_metrics_list])),
            "std": float(np.std([m["f1"] for m in fold_metrics_list])),
            "values": [m["f1"] for m in fold_metrics_list]
        },
        "combined_accuracy": float(accuracy_score(all_y_true, all_y_pred))
    }
    
    print(f"\nCross-Validation Results (Mean ± Std):")
    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        mean = aggregated_metrics[metric_name]["mean"]
        std = aggregated_metrics[metric_name]["std"]
        print(f"  {metric_name}: {mean:.4f} ± {std:.4f}")
    print(f"  combined_accuracy: {aggregated_metrics['combined_accuracy']:.4f}")
    
    # Save aggregated results
    with open(run_dir / "cv_results.json", "w") as f:
        json.dump(aggregated_metrics, f, indent=2)
    
    with open(run_dir / "fold_metrics.json", "w") as f:
        json.dump(fold_metrics_list, f, indent=2)
    
    # Save combined confusion matrix
    combined_cm = confusion_matrix(all_y_true, all_y_pred)
    np.save(run_dir / "confusion_matrix.npy", combined_cm)
    
    combined_cm_labeled = {
        "labels": class_names,
        "matrix": combined_cm.tolist(),
        "description": "Combined confusion matrix aggregated from all fold predictions"
    }
    with open(run_dir / "confusion_matrix.json", "w") as f:
        json.dump(combined_cm_labeled, f, indent=2)
    
    # Save average confusion matrix
    avg_cm = np.mean(fold_cms, axis=0)
    np.save(run_dir / "avg_confusion_matrix.npy", avg_cm)
    
    avg_cm_labeled = {
        "labels": class_names,
        "matrix": avg_cm.tolist(),
        "description": "Average of individual fold confusion matrices"
    }
    with open(run_dir / "avg_confusion_matrix.json", "w") as f:
        json.dump(avg_cm_labeled, f, indent=2)
    
    print(f"\n+ All results saved to: {run_dir}")
    
    return run_dir, aggregated_metrics


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.training.train_temporal <config_path>")
        print("Example: python -m src.training.train_temporal configs/temporal_trial1.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path)
