"""
Local equivalent of the GPU Detection & Environment Setup
step, adapted for the local RTX 5070 Ti setup instead of Kaggle's dual-GPU
environment.

Checks that the local machine is ready for model training.

Verifies:
- Python version
- TensorFlow installation
- PyTorch CUDA support
- GPU detection
- Required libraries

Run:
    python scripts/verify_environment.py
"""

import sys


def check_python() -> None:
    print(f"Python: {sys.version.split()[0]}")


def check_tensorflow() -> None:
    try:
        import tensorflow as tf

        print(f"TensorFlow: {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        print(f"  TensorFlow GPUs detected: {len(gpus)} (expected 0 -- CPU-only on Windows)")
    except ImportError:
        print("TensorFlow: NOT INSTALLED")


def check_pytorch() -> None:
    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_available}")
        if cuda_available:
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  [WARNING] CUDA not available -- BERT training needs GPU acceleration")
    except ImportError:
        print("PyTorch: NOT INSTALLED")


def check_other_packages() -> None:
    packages = ["pandas", "numpy", "sklearn", "transformers", "streamlit", "google.generativeai"]
    print("\nOther key packages:")
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "version unknown")
            print(f"  {pkg}: {version}")
        except ImportError:
            print(f"  {pkg}: NOT INSTALLED")


def main() -> None:
    print("== Environment Verification (local, RTX 5070 Ti) ==\n")
    check_python()
    check_tensorflow()
    check_pytorch()
    check_other_packages()
    print("\nIf TensorFlow shows 0 GPUs, that's expected -- TF stays CPU-only on")
    print("native Windows (GPU support dropped after v2.10). BiLSTM trains fine on CPU.")
    print("PyTorch should show CUDA available with local RTX 5070 Ti for BERT training.")


if __name__ == "__main__":
    main()
