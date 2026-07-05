# SETUP.md

# Emotion Detection & Learning Support Engine

This file contains the setup steps I used while developing the project. If you're setting up the project on a new machine, follow these steps.

---

## Python Version

This project was developed using **Python 3.12.10**.

Using Python 3.14 caused compatibility issues with TensorFlow 2.16.1.

Check your version:

```bash
python --version
```

---

## Clone the Repository

```bash
git clone https://github.com/weeb-with-net/emotion-detection-engine-lsa.git emotion-learning-assistant-repo

cd emotion-learning-assistant-repo
```
---

## Create a Virtual Environment

```bash
py -3.12 -m venv .venv
```

Activate it:

### PowerShell / VS Code Terminal

```powershell
.\.venv\Scripts\Activate.ps1
```

### Command Prompt

```cmd
.venv\Scripts\activate.bat
```

---

## Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## Install PyTorch (GPU)

If you have an NVIDIA GPU, install PyTorch separately before installing the other packages.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

During testing, using `--index-url` worked correctly on RTX 5070 Ti and RTX 5060 systems. Using `--extra-index-url` sometimes installed the CPU version of PyTorch instead.

---

## Install Remaining Packages

```bash
pip install -r requirements.txt
```

---

## Check TensorFlow

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

Expected output:

```
2.16.1
```

---

## Check PyTorch

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If everything is installed correctly, CUDA should be available.

To see which GPU PyTorch is using:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Notes

- TensorFlow is used for the BiLSTM model.
- PyTorch is used for the BERT model.
- TensorFlow is intentionally installed as the CPU version because newer TensorFlow releases don't support native Windows GPU training. The BiLSTM model is lightweight, so CPU training is sufficient.
- GPU acceleration is used only for the PyTorch models.

---

## If Something Doesn't Work

### TensorFlow installation fails

Make sure you're using Python 3.12.

```bash
python --version
```

---

### PyTorch doesn't detect the GPU

Try reinstalling it:

```bash
pip uninstall torch torchvision torchaudio

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

You can also check that Windows detects your GPU by running:

```bash
nvidia-smi
```

---

Last tested on:

- Windows 11
- Python 3.12.10
- TensorFlow 2.16.1
- PyTorch 2.11.0 + CUDA 12.8