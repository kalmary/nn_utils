![BANNER](https://github.com/kalmary/nn_utils/blob/main/img/BANNER_NN_UTILS.png)


---

#### **nn_utils** is a simple, modular collection of utilities designed to simplify the process of building, debugging, and experimenting with neural networks in Python.

---

### Code example:


![CODE](https://github.com/kalmary/nn_utils/blob/main/img/code_example.png)

---

### Instalation: 

Clone the repository to your local machine:

```bash
git clone [https://github.com/kalmary/nn_utils.git]
cd nn_utils
```
---

### File structure:

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
