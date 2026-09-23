![Language](https://img.shields.io/badge/Language-Python-green) ![OS](https://img.shields.io/badge/OS-Linux,Windows,macOS-yellow)

---

### Overview

**nn_utils** is a simple, modular collection of utilities designed to simplify the process of nn models' evaluation and training's observation in Python.

---

### Instalation: 

Clone the repository to your local machine:

```bash
git clone [https://github.com/kalmary/nn_utils.git]
cd nn_utils

uv sync --extra pytorch-cpu  # core runtime, CPU-only
uv sync --group test --extra pytorch-cpu  # CPU-only on macOS, Windows, or Linux
# macOS system profile (also CPU): uv sync --group test --extra pytorch-macos
# Linux with CUDA 13.2: uv sync --group test --extra pytorch-linux-cuda
# Windows with CUDA 13.2: uv sync --group test --extra pytorch-windows-cuda
```


---

### Usage

Run imports and utilities through the locked environment using the same PyTorch profile selected during installation:

```bash
uv run --no-sync python -c "from src.file_handling import load_json"
uv run --no-sync pytest
```

When this repository is used as the `nn_utils` package inside a parent project, import its public API with `from nn_utils import ...`; plotting names are loaded only when requested.

---

### Implemented functions and files structure:

```
src
├── accuracy_metrics.py
│   ├── get_Probabilities
│   ├── get_intLabels
│   ├── calculate_accuracy
│   ├── calculate_weighted_accuracy
│   ├── compute_mIoU
│   ├── get_dataset_len
│   └── calculate_class_weights
├── evaluation_plot_tools.py
│   └── Plotter
│   │   ├── plot_metric_hist
│   │   ├── plot_metric_hist
│   │   ├── cnf_matrix
│   │   ├── cnf_matrix_analysis
│   │   ├── prc_curve
│   │   └── roc_curve
│   └── ClassificationReport
├── file_handling.py
│   ├── convert_str_values
│   ├── save_model
│   ├── load_model
│   ├── save2json
│   └── load_json
├── loss_functions.py
│   ├── IoULoss
│   ├── FocalLoss_ArcFace
│   ├── DiceLoss
│   ├── FocalLoss
│   └── LabelSmoothingFocalLoss
└── training_callbacks.py
    └── EarlyStopping
```

### **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
