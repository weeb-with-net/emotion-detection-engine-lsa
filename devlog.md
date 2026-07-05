# Emotion Detection & Learning Support Engine

## Development Log

**Project Duration:** July 2026 (Current Progress)

------------------------------------------------------------------------

# Project Initialization

-   Established the overall project architecture before implementation.
-   Broke development into Epics and Tasks for incremental progress.
-   Adopted a structured software engineering workflow.

## Initial Architecture

-   Streamlit frontend
-   TensorFlow BiLSTM model
-   PyTorch BERT model
-   Gemini API integration
-   Local model inference

------------------------------------------------------------------------

# Epic 1 --- Development Environment

## Environment Setup

Successfully configured the complete local development environment.

### Major Decisions

-   Created a dedicated Python virtual environment.
-   Downgraded Python from 3.14 to 3.12 for ML compatibility.
-   Installed TensorFlow (CPU).
-   Installed CUDA-enabled PyTorch (cu128).
-   Verified NVIDIA RTX 5070 Ti GPU acceleration.

### Environment Validation

Verified:

-   CUDA available
-   PyTorch detects GPU
-   TensorFlow functioning correctly
-   Stable TensorFlow/PyTorch coexistence

**Engineering Decision**

> Never revisit environment setup unless something breaks.

------------------------------------------------------------------------

# Local Training Strategy

The original workflow suggested Kaggle notebooks.

After evaluation, the workflow was intentionally changed to local
training.

### Reasons

-   Faster iteration on RTX 5070 Ti
-   Easier debugging
-   No cloud notebook limitations
-   Better reproducibility
-   Simpler workflow

------------------------------------------------------------------------

# Git Workflow

Adopted incremental commits after each completed task instead of
milestone commits.

Planned history:

-   Epic 1 complete
-   Epic 2 T1
-   Epic 2 T2
-   Epic 2 T3
-   ...
-   Documentation

------------------------------------------------------------------------

# Epic 2 --- Dataset Preparation

## Dataset Selection

Selected datasets:

-   GoEmotions
-   EmpatheticDialogues
-   ISEAR

Organized project structure:

``` text
data/
├── raw/
└── processed/
```

Configured `.gitignore` to exclude raw datasets.

------------------------------------------------------------------------

# Dataset Analysis

Performed exploratory analysis of class distribution.

Observed significant imbalance, particularly for the **Bored** class.

## Decision

Instead of:

-   Oversampling
-   Synthetic sample generation
-   Removing the class

The project will use **class weighting** during model training.

This preserves all classes while minimizing overfitting risk.

------------------------------------------------------------------------

# Model Training Plan

## Model 1

-   BiLSTM
-   TensorFlow

## Model 2

-   BERT
-   PyTorch
-   GPU fine-tuning

Both models will later be integrated into the Streamlit application.

------------------------------------------------------------------------

# Initial Hyperparameters

``` text
Batch Size              : 16
Maximum Epochs          : 50
Early Stopping Patience : 10
```

Target performance:

-   Accuracy ≈ 85%
-   Strong Macro F1
-   Good generalization

Potential future tuning:

-   Learning-rate scheduler
-   Weight decay
-   Dropout optimization
-   Sequence length tuning
-   Class weighting
-   Hyperparameter search

------------------------------------------------------------------------

# Engineering Principles

-   Keep the architecture simple.
-   Prefer reproducibility.
-   Train locally whenever possible.
-   Commit small, meaningful changes.
-   Validate engineering decisions before implementation.
-   Minimize unnecessary deviations from the project specification.

------------------------------------------------------------------------

# Current Status

## ✅ Completed

-   Project planning
-   Architecture design
-   Epic 1
-   Environment setup
-   CUDA configuration
-   Local training strategy
-   Dataset download
-   Dataset organization
-   Dataset exploration
-   Class imbalance analysis
-   Git workflow

## 🚧 In Progress

-   Epic 2
-   Data preprocessing pipeline

## ⏳ Upcoming

-   Text preprocessing
-   Feature engineering
-   BiLSTM training
-   BERT fine-tuning
-   Model evaluation
-   Streamlit integration
-   Gemini API integration
-   Documentation
-   Final testing
-   Project submission

------------------------------------------------------------------------

# Key Lessons Learned

-   Proper environment setup saves substantial debugging time later.
-   Local GPU training provides a significantly faster iteration cycle
    than cloud notebooks.
-   Class imbalance is best handled during training using class
    weighting.
-   Incremental Git commits improve traceability and maintenance.
-   Early planning reduces implementation uncertainty.

------------------------------------------------------------------------

*End of current development log.*
