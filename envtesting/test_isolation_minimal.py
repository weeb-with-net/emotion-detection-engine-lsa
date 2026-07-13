"""
Minimal test for the TF/PyTorch segfault - just imports, no models
loaded. Per scripts/_integration_check_bilstm.py, the old finding was
that just importing both in the same process crashes it, without
either model ever being touched. This checks if that's still true with
your current pinned versions.

Run from the project root:
    python test_isolation_minimal.py

Watch for: does "ALL IMPORTS OK" print at the end? If the process dies
silently, shows "Segmentation fault", or the terminal just closes/shows
an access-violation exit code with no final print - that's the crash.
"""
print("Step 1: importing tensorflow...")
import tensorflow as tf
print(f"  OK - tensorflow {tf.__version__}")

print("Step 2: importing torch...")
import torch
print(f"  OK - torch {torch.__version__}")

print("Step 3: importing transformers...")
import transformers
print(f"  OK - transformers {transformers.__version__}")

print("\nALL IMPORTS OK - no crash from import alone.")
