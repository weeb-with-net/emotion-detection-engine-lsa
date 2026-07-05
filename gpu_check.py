"""
Quick sanity check: confirms PyTorch is installed correctly and can see
your GPU. Run this once, right after `pip install -r requirements.txt`,
before moving on to any model work.

    python gpu_check.py
"""

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU name:        {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print("\n✅ GPU is ready for BERT training/inference.")
else:
    print(
        "\n⚠️  PyTorch can't see the GPU."
    )
