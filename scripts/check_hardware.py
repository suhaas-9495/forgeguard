#!/usr/bin/env python3
"""Hardware verification script for ForgeGuard."""

import sys
import subprocess

def check_hardware():
    print("=" * 60)
    print("  ForgeGuard — Hardware Verification")
    print("=" * 60)

    # Python version
    print(f"\n✅ Python: {sys.version.split()[0]}")

    # PyTorch + CUDA
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            vram_gb = gpu.total_memory / 1024**3
            print(f"✅ CUDA: {torch.version.cuda}")
            print(f"✅ GPU: {gpu.name}")
            print(f"✅ VRAM: {vram_gb:.1f} GB")
            if vram_gb < 5.5:
                print("⚠️  WARNING: Less than 6GB VRAM — reduce batch size")
            else:
                print(f"✅ VRAM sufficient for QLoRA training")
        else:
            print("❌ CUDA not available — training will be slow on CPU")
    except ImportError:
        print("❌ PyTorch not installed — run: pip install torch")

    # Key packages
    packages = [
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("bitsandbytes", "bitsandbytes"),
        ("accelerate", "accelerate"),
        ("trl", "trl"),
        ("datasets", "datasets"),
        ("gradio", "gradio"),
    ]

    print("\n--- Package Check ---")
    all_ok = True
    for name, module in packages:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "installed")
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: NOT INSTALLED")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("🚀 All systems ready! You can start training.")
    else:
        print("⚠️  Install missing packages: pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    check_hardware()
