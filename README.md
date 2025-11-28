![BANNER](https://github.com/kalmary/nn_utils/blob/main/img/BANNER_NN_UTILS.png)
![Language](https://img.shields.io/badge/Language-Python-green) ![OS](https://img.shields.io/badge/OS-Linux,Windows-yellow)

---

### Overview

**nn_utils** is a simple, modular collection of utilities designed to simplify the process of building, debugging, and experimenting with neural networks in Python.

---

### Instalation: 

Clone the repository to your local machine:

```bash
git clone [https://github.com/kalmary/nn_utils.git]
cd nn_utils
```

---

### Code example:

| CODE | OUTPUT (TESTOWENARAZIE) |
| :---: | :---: |
| ![CODE](https://github.com/kalmary/nn_utils/blob/main/img/code_example.png) | ![CODE1](https://github.com/kalmary/nn_utils/blob/main/img/obraz.png) |



---



### File structure:

```
nn_utils
├── img
├── src
│   ├── accuracy_metrics.py
│   ├── evaluation_plot_tools.py
│   ├── file_handling.py
│   ├── loss_functions.py
│   └── training_callbacks.py
├── .gitattributes
├── .gitignore
├── Makefile
├── README.md
└── __init__.py
```


---

### Implemented functions:

```
src
├── accuracy_metrics.py
│   ├── get_Probabilities
│   ├── get_intLabels
│   ├── calculate_accuracy
│   ├── calculate_weighted_accuracy
│   ├── get_dataset_len
│   ├── calculate_class_weights
│   └── compute_mIoU
├── evaluation_plot_tools.py
│   ├── plot_loss
│   ├── plot_accuracy
│   ├── cnf_matrix
│   ├── cnf_matrix_analysis
│   ├── prc_curve
│   ├── roc_curve
│   └── ClassificationReport
├── file_handling.py
│   ├── convert_str_values
│   ├── _existing_files
│   ├── save_model
│   ├── load_model
│   ├── save2json
│   └── load_json
├── loss_functions.py
│   ├── IoULoss
│   ├── DiceLoss
│   ├── FocalLoss
│   └── LabelSmoothingFocalLoss
└── training_callbacks.py
    └── *EarlyStopping
```


* accuracy_metrics.py
  * get_Probabilities
  * get_intLabels
  * calculate_accuracy
  * calculate_weighted_accuracy
  * get_dataset_len
  * calculate_class_weights
  * compute_mIoU
* evaluation_plot_tools.py
  * plot_loss
  * plot_accuracy
  * cnf_matrix
  * cnf_matrix_analysis
  * prc_curve
  * roc_curve
  * ClassificationReport
* file_handling.py
  * convert_str_values
  * _existing_files
  * save_model
  * load_model
  * save2json
  * load_json
* loss_functions.py
  * IoULoss
  * DiceLoss
  * FocalLoss
  * LabelSmoothingFocalLoss
* training_callbacks.py
  * EarlyStopping
