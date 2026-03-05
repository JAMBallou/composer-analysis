"""
gpu_config.py
-------------------------
GPU configuration utilities for TensorFlow/Keras training.

Optimizes GPU memory usage and enables performance features like:
- Dynamic memory growth (prevents OOM errors)
- Mixed precision training (faster on modern GPUs)
- Multi-GPU support
- GPU device visibility control
"""

import os
import tensorflow as tf
from typing import Optional, List


def configure_gpu(
    memory_growth: bool = True,
    mixed_precision: bool = True,
    gpu_ids: Optional[List[int]] = None,
    memory_limit_mb: Optional[int] = None,
    log_device_placement: bool = False,
    verbose: bool = True
) -> dict:
    """
    Configure TensorFlow GPU settings for optimal performance.
    
    Args:
        memory_growth: Enable dynamic GPU memory allocation (recommended)
        mixed_precision: Enable mixed precision (float16/float32) training
        gpu_ids: List of GPU IDs to use (None = use all available)
        memory_limit_mb: Maximum GPU memory to use in MB (None = no limit)
        log_device_placement: Log which operations run on which device
        verbose: Print configuration details
    
    Returns:
        dict: Configuration summary with GPU info
    """
    config_summary = {
        'gpus_available': [],
        'gpus_configured': [],
        'memory_growth': memory_growth,
        'mixed_precision': False,
        'memory_limit_mb': memory_limit_mb,
        'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
    }
    
    # ===== Check GPU Availability =====
    try:
        gpus = tf.config.list_physical_devices('GPU')
        config_summary['gpus_available'] = [gpu.name for gpu in gpus]
        
        if not gpus:
            if verbose:
                print("⚠️  No GPUs detected - training will run on CPU")
                print("   For GPU support, ensure CUDA and cuDNN are installed")
            return config_summary
        
        if verbose:
            print(f"🎮 Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   [{i}] {gpu.name}")
        
        # ===== Set Visible GPUs =====
        if gpu_ids is not None:
            # Filter to specific GPUs
            selected_gpus = [gpus[i] for i in gpu_ids if i < len(gpus)]
            tf.config.set_visible_devices(selected_gpus, 'GPU')
            gpus = selected_gpus
            if verbose:
                print(f"   Using GPU(s): {gpu_ids}")
        
        # ===== Configure Memory Growth =====
        if memory_growth:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    if verbose:
                        print(f"✓ Memory growth enabled for {gpu.name}")
                except RuntimeError as e:
                    # Memory growth must be set before GPUs are initialized
                    if verbose:
                        print(f"⚠️  Could not set memory growth (GPU already initialized): {e}")
        
        # ===== Set Memory Limit =====
        if memory_limit_mb is not None:
            for gpu in gpus:
                try:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                    )
                    if verbose:
                        print(f"✓ Memory limit set to {memory_limit_mb} MB for {gpu.name}")
                except RuntimeError as e:
                    if verbose:
                        print(f"⚠️  Could not set memory limit: {e}")
        
        config_summary['gpus_configured'] = [gpu.name for gpu in gpus]
        
    except Exception as e:
        if verbose:
            print(f"⚠️  GPU configuration error: {e}")
        return config_summary
    
    # ===== Enable Mixed Precision =====
    if mixed_precision and gpus:
        try:
            # Mixed precision is beneficial for GPUs with Tensor Cores
            # (RTX 20xx, 30xx, 40xx, 50xx series)
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            config_summary['mixed_precision'] = True
            
            if verbose:
                print(f"✓ Mixed precision (float16) enabled")
                print(f"   → Faster training on RTX GPUs with Tensor Cores")
                print(f"   → Compute dtype: {policy.compute_dtype}")
                print(f"   → Variable dtype: {policy.variable_dtype}")
        except Exception as e:
            if verbose:
                print(f"⚠️  Could not enable mixed precision: {e}")
    elif not gpus and mixed_precision:
        if verbose:
            print("⚠️  Mixed precision disabled - CPU training is faster with float32")
    
    # ===== Log Device Placement =====
    if log_device_placement:
        tf.debugging.set_log_device_placement(True)
        if verbose:
            print("✓ Device placement logging enabled")
    
    # ===== Print Summary =====
    if verbose:
        print("\n" + "="*60)
        print("GPU Configuration Summary")
        print("="*60)
        print(f"GPUs Available:    {len(config_summary['gpus_available'])}")
        print(f"GPUs Configured:   {len(config_summary['gpus_configured'])}")
        print(f"Memory Growth:     {config_summary['memory_growth']}")
        print(f"Mixed Precision:   {config_summary['mixed_precision']}")
        if memory_limit_mb:
            print(f"Memory Limit:      {memory_limit_mb} MB")
        print("="*60 + "\n")
    
    return config_summary


def get_gpu_info() -> dict:
    """
    Get detailed information about available GPUs.
    
    Returns:
        dict: GPU information including CUDA/cuDNN versions
    """
    info = {
        'cuda_available': tf.test.is_built_with_cuda(),
        'gpu_available': tf.test.is_gpu_available(),
        'cuda_version': None,
        'cudnn_version': None,
        'gpu_devices': []
    }
    
    # Get CUDA/cuDNN versions
    try:
        info['cuda_version'] = tf.sysconfig.get_build_info().get('cuda_version', 'Unknown')
        info['cudnn_version'] = tf.sysconfig.get_build_info().get('cudnn_version', 'Unknown')
    except:
        pass
    
    # Get GPU device details
    gpus = tf.config.list_physical_devices('GPU')
    for i, gpu in enumerate(gpus):
        gpu_detail = {
            'id': i,
            'name': gpu.name,
            'device_type': gpu.device_type
        }
        
        # Try to get GPU memory info (requires nvidia-ml-py3)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_detail['gpu_name'] = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_detail['memory_total_gb'] = round(mem_info.total / 1024**3, 2)
            gpu_detail['memory_free_gb'] = round(mem_info.free / 1024**3, 2)
            gpu_detail['memory_used_gb'] = round(mem_info.used / 1024**3, 2)
            pynvml.nvmlShutdown()
        except:
            pass
        
        info['gpu_devices'].append(gpu_detail)
    
    return info


def print_gpu_info():
    """Print detailed GPU information for diagnostics."""
    info = get_gpu_info()
    
    print("\n" + "="*60)
    print("GPU System Information")
    print("="*60)
    print(f"TensorFlow version:  {tf.__version__}")
    print(f"Built with CUDA:     {info['cuda_available']}")
    print(f"GPU available:       {info['gpu_available']}")
    
    if info['cuda_version']:
        print(f"CUDA version:        {info['cuda_version']}")
    if info['cudnn_version']:
        print(f"cuDNN version:       {info['cudnn_version']}")
    
    print(f"\nDetected GPUs:       {len(info['gpu_devices'])}")
    for gpu in info['gpu_devices']:
        print(f"\n  GPU {gpu['id']}: {gpu.get('gpu_name', gpu['name'])}")
        if 'memory_total_gb' in gpu:
            print(f"    Total Memory:  {gpu['memory_total_gb']} GB")
            print(f"    Free Memory:   {gpu['memory_free_gb']} GB")
            print(f"    Used Memory:   {gpu['memory_used_gb']} GB")
    
    print("="*60 + "\n")


def auto_configure_gpu(verbose: bool = True) -> dict:
    """
    Automatically configure GPU with recommended settings.
    
    This is the recommended configuration for most use cases:
    - Enables memory growth (prevents OOM errors)
    - Enables mixed precision on RTX GPUs
    - Uses all available GPUs
    
    Args:
        verbose: Print configuration details
    
    Returns:
        dict: Configuration summary
    """
    return configure_gpu(
        memory_growth=True,
        mixed_precision=True,
        gpu_ids=None,  # Use all GPUs
        memory_limit_mb=None,  # No hard limit
        log_device_placement=False,
        verbose=verbose
    )


# ===== Convenience function =====
def setup_gpu(verbose: bool = True) -> dict:
    """Alias for auto_configure_gpu()."""
    return auto_configure_gpu(verbose=verbose)


if __name__ == "__main__":
    # Test configuration
    print("Testing GPU configuration...\n")
    print_gpu_info()
    config = auto_configure_gpu(verbose=True)
    
    # Simple GPU test
    if config['gpus_configured']:
        print("\n🧪 Running simple GPU test...")
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
        print("✓ GPU computation successful!")
    else:
        print("\n⚠️  No GPU configured - skipping GPU test")
