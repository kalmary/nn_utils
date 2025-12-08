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

python -m venv .venv
source .venv/bin/activate

pip install requirements.txt # install all requirements, without pytorch and cuda

# tested on this, but should work with any other version
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 
```


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
