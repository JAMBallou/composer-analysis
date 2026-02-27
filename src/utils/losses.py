"""
losses.py
---------
Custom loss functions for handling class imbalance in composer classification.

Implements:
- Weighted categorical crossentropy (using class weights)
- Focal loss (focuses on hard examples, reduces impact of easy examples)
- Balanced variants that adjust for class frequency
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Dict


def weighted_sparse_categorical_crossentropy(class_weights: Optional[Dict[int, float]] = None):
    """
    Create a weighted sparse categorical crossentropy loss function.
    
    This loss function weights each class inversely proportional to its frequency,
    giving more importance to minority classes during training.
    
    Args:
        class_weights: Dict mapping class indices to weight scalars.
                      If None, all classes are weighted equally.
    
    Returns:
        A loss function that can be passed to model.compile()
    
    Example:
        >>> class_weights = {0: 0.5, 1: 1.0, 2: 2.0}
        >>> loss_fn = weighted_sparse_categorical_crossentropy(class_weights)
        >>> model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    """
    
    def loss(y_true, y_pred):
        # Cast labels to int32 for sparse_softmax_cross_entropy_with_logits
        y_true_int = tf.cast(y_true, tf.int32)
        
        # Get unweighted crossentropy loss
        base_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_int, logits=y_pred)
        
        if class_weights is None:
            return tf.reduce_mean(base_loss)
        
        # Apply class weights
        weights = tf.gather(
            tf.constant([class_weights.get(i, 1.0) for i in range(len(class_weights))]),
            y_true_int
        )
        weighted_loss = base_loss * weights
        
        return tf.reduce_mean(weighted_loss)
    
    return loss


def focal_loss(alpha: float = 0.25, gamma: float = 2.0, class_weights: Optional[Dict[int, float]] = None):
    """
    Focal loss for addressing class imbalance (from RetinaNet paper).
    
    Focal loss applies a modulating term to the cross entropy loss to focus learning
    on hard negative examples. This is particularly useful for imbalanced datasets.
    
    The loss is: Loss = -alpha * (1 - pt)^gamma * log(pt)
    where pt is the probability of the ground-truth class.
    
    Args:
        alpha: Weighting factor in [0, 1] to balance positive vs negative examples.
               Higher alpha = more weight on positive class. Default: 0.25
        gamma: Focusing parameter for modulating loss from hard examples.
               gamma=0 is standard cross entropy. gamma=2 is standard focal loss. Default: 2.0
        class_weights: Additional per-class weights. If provided, combines with focal loss.
    
    Returns:
        A loss function that can be passed to model.compile()
    
    Example:
        >>> loss_fn = focal_loss(alpha=0.25, gamma=2.0)
        >>> model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    """
    
    def loss(y_true, y_pred):
        # Convert labels to one-hot for easier computation
        y_true_int = tf.cast(y_true, tf.int32)
        
        # Compute softmax probabilities
        predictions = tf.nn.softmax(y_pred)
        
        # Get probability of ground truth class
        epsilon = tf.keras.backend.epsilon()
        predictions = tf.clip_by_value(predictions, epsilon, 1.0 - epsilon)
        
        # Compute crossentropy
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_int, logits=y_pred)
        
        # Get probabilities of true class
        batch_size = tf.shape(y_pred)[0]
        batch_indices = tf.range(batch_size)
        true_indices = tf.stack([batch_indices, y_true_int], axis=1)
        pt = tf.gather_nd(predictions, true_indices)
        
        # Compute focal weight: (1 - pt)^gamma
        focal_weight = tf.pow(1.0 - pt, gamma)
        
        # Apply focal loss
        focal_loss_value = alpha * focal_weight * ce_loss
        
        # Apply class weights if provided
        if class_weights is not None:
            weights = tf.gather(
                tf.constant([class_weights.get(i, 1.0) for i in range(len(class_weights))]),
                y_true_int
            )
            focal_loss_value = focal_loss_value * weights
        
        return tf.reduce_mean(focal_loss_value)
    
    return loss


def balanced_weighted_loss(
    class_weights: Optional[Dict[int, float]] = None,
    loss_type: str = "weighted_crossentropy"
):
    """
    Factory function to create a balanced loss function with class weights.
    
    Args:
        class_weights: Dict mapping class indices to weight scalars.
        loss_type: Type of loss to use.
                  Options: 'weighted_crossentropy', 'focal_loss'
    
    Returns:
        A loss function suitable for model.compile()
    """
    if loss_type == "weighted_crossentropy":
        return weighted_sparse_categorical_crossentropy(class_weights)
    elif loss_type == "focal_loss":
        return focal_loss(class_weights=class_weights)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def compute_class_weights_balanced(y_train: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights: weight = n_samples / (n_classes * n_samples_per_class).
    
    This ensures that samples of each class contribute equally to the total loss,
    regardless of their frequency in the training set.
    
    Args:
        y_train: Array of training labels
    
    Returns:
        Dictionary mapping class indices to weight scalars
    
    Example:
        >>> y_train = np.array([0, 0, 0, 1, 2, 2])
        >>> weights = compute_class_weights_balanced(y_train)
        >>> print(weights)  # {0: 0.5, 1: 1.0, 2: 0.66...}
    """
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    n_samples = len(y_train)
    n_classes = len(unique_classes)
    
    weights = {}
    for cls, count in zip(unique_classes, class_counts):
        weight = n_samples / (n_classes * count)
        weights[int(cls)] = float(weight)
    
    return weights


def compute_class_weights_inverse_frequency(y_train: np.ndarray) -> Dict[int, float]:
    """
    Compute inverse frequency class weights: weight = 1 / frequency.
    
    Simpler approach: minority classes get higher weights.
    
    Args:
        y_train: Array of training labels
    
    Returns:
        Dictionary mapping class indices to weight scalars
    """
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    
    weights = {}
    for cls, count in zip(unique_classes, class_counts):
        frequency = count / total_samples
        weight = 1.0 / frequency
        weights[int(cls)] = float(weight)
    
    # Normalize so average weight is 1.0
    avg_weight = np.mean(list(weights.values()))
    weights = {k: v / avg_weight for k, v in weights.items()}
    
    return weights


if __name__ == "__main__":
    # Test the loss functions
    import tensorflow as tf
    
    print("Testing loss functions...\n")
    
    # Create dummy data
    y_true = tf.constant([0, 0, 0, 1, 2, 2])
    y_pred = tf.random.normal((6, 3))
    
    # Test balanced weights
    weights = compute_class_weights_balanced(y_true.numpy())
    print(f"Balanced weights: {weights}")
    
    # Test inverse frequency weights
    weights = compute_class_weights_inverse_frequency(y_true.numpy())
    print(f"Inverse frequency weights: {weights}\n")
    
    # Test loss functions
    loss_fn = weighted_sparse_categorical_crossentropy(weights)
    loss_value = loss_fn(y_true, y_pred)
    print(f"Weighted crossentropy loss: {loss_value.numpy():.4f}")
    
    loss_fn = focal_loss(alpha=0.25, gamma=2.0, class_weights=weights)
    loss_value = loss_fn(y_true, y_pred)
    print(f"Focal loss: {loss_value.numpy():.4f}")
