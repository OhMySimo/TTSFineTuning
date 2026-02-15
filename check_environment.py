#!/usr/bin/env python3
"""
Quick environment check for Qwen3-TTS training on Vast.ai
Run this before starting training to ensure everything is set up correctly
"""

import sys
import os

def check_mark(condition, message):
    """Print check mark or X based on condition"""
    symbol = "✓" if condition else "❌"
    status = "OK" if condition else "FAIL"
    print(f"{symbol} {message}: {status}")
    return condition

def main():
    print("=" * 60)
    print("🔍 Qwen3-TTS Environment Check")
    print("=" * 60)
    print()
    
    all_checks_passed = True
    
    # Check Python version
    print("📋 Python Environment:")
    py_version = sys.version_info
    py_ok = py_version.major == 3 and py_version.minor >= 8
    all_checks_passed &= check_mark(py_ok, f"Python {py_version.major}.{py_version.minor}")
    print()
    
    # Check GPU
    print("🎮 GPU Check:")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        all_checks_passed &= check_mark(cuda_available, "CUDA available")
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            all_checks_passed &= check_mark(gpu_count >= 1, f"GPU count: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  • GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Check if RTX 3090
            is_3090 = "3090" in torch.cuda.get_device_name(0)
            if is_3090:
                print(f"  💡 RTX 3090 detected - optimal configuration enabled")
        else:
            print("  ⚠️  No GPU detected - training will be very slow!")
            
    except ImportError:
        all_checks_passed &= check_mark(False, "PyTorch")
    print()
    
    # Check required packages
    print("📦 Required Packages:")
    packages = {
        'qwen_tts': 'qwen-tts',
        'accelerate': 'accelerate',
        'soundfile': 'soundfile',
        'tensorboard': 'tensorboard',
    }
    
    for module, package in packages.items():
        try:
            __import__(module)
            check_mark(True, package)
        except ImportError:
            all_checks_passed &= check_mark(False, f"{package} (missing)")
    print()
    
    # Check optional packages
    print("🔧 Optional Packages (for performance):")
    optional = {
        'flash_attn': 'flash-attention',
        'bitsandbytes': 'bitsandbytes (8-bit Adam)',
    }
    
    for module, package in optional.items():
        try:
            __import__(module)
            check_mark(True, package)
        except ImportError:
            print(f"  ⚠️  {package}: not installed (recommended)")
    print()
    
    # Check files
    print("📁 Training Files:")
    files = {
        'sft_12hz.py': 'Training script',
        'dataset.py': 'Dataset loader',
        'prepare_data.py': 'Data preparation',
        'train_vast.sh': 'Launch script',
    }
    
    for filename, description in files.items():
        exists = os.path.exists(filename)
        all_checks_passed &= check_mark(exists, f"{description} ({filename})")
    print()
    
    # Check dataset
    print("📊 Dataset:")
    dataset_file = 'train_raw.jsonl'
    dataset_exists = os.path.exists(dataset_file)
    
    if dataset_exists:
        try:
            with open(dataset_file) as f:
                num_lines = sum(1 for _ in f)
            check_mark(True, f"Dataset found ({num_lines} samples)")
            
            # Quick validation
            with open(dataset_file) as f:
                import json
                first_line = json.loads(f.readline())
                has_audio = 'audio' in first_line
                has_text = 'text' in first_line
                has_ref = 'ref_audio' in first_line
                
                all_checks_passed &= check_mark(has_audio, "  'audio' field present")
                all_checks_passed &= check_mark(has_text, "  'text' field present")
                all_checks_passed &= check_mark(has_ref, "  'ref_audio' field present")
        except Exception as e:
            all_checks_passed &= check_mark(False, f"Dataset validation: {str(e)}")
    else:
        print(f"  ⚠️  {dataset_file} not found - upload your dataset!")
        all_checks_passed = False
    print()
    
    # Check disk space
    print("💾 Storage:")
    try:
        import shutil
        total, used, free = shutil.disk_usage("/workspace")
        free_gb = free / (1024**3)
        space_ok = free_gb >= 50
        all_checks_passed &= check_mark(space_ok, f"Free disk space: {free_gb:.1f}GB")
        
        if not space_ok:
            print(f"  ⚠️  Warning: Need at least 50GB free (checkpoints ~3.5GB each)")
    except:
        print("  ⚠️  Could not check disk space")
    print()
    
    # Check models download
    print("🤖 Model Cache:")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    models_cached = os.path.exists(cache_dir) and os.listdir(cache_dir)
    
    if models_cached:
        print("  ✓ HuggingFace cache found (models may be pre-downloaded)")
    else:
        print("  ℹ️  Models will be downloaded on first use (~4GB)")
    print()
    
    # Final summary
    print("=" * 60)
    if all_checks_passed:
        print("✅ All critical checks passed!")
        print("🚀 Ready to start training")
        print()
        print("Next steps:")
        print("  1. chmod +x train_vast.sh")
        print("  2. ./train_vast.sh")
        print("=" * 60)
        return 0
    else:
        print("❌ Some checks failed - please fix issues above")
        print()
        print("Common fixes:")
        print("  • Missing packages: pip install --break-system-packages <package>")
        print("  • Missing files: scp files from your local machine")
        print("  • Missing dataset: upload train_raw.jsonl")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
