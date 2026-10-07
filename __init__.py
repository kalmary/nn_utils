from .src.accuracy_metrics import (
    calculate_accuracy,
    calculate_weighted_accuracy,
    compute_miou,
    compute_pos_weights,
    compute_pos_weights_cloud,
    compute_pos_weights_h5,
    get_dataset_len,
    get_int_labels,
    get_probabilities,
)
from .src.file_handling import (
    convert_str_values,
    load_json,
    load_model,
    save2json,
    save_model,
    wrap_hist,
)
from .src.loss_functions import (
    ArcfaceFocalLoss,
    DiceLoss,
    DiscriminativeLoss,
    FocalLoss,
    IouLoss,
    calculate_l1_penalty_best_practice,
)
from .src.training_callbacks import EarlyStopping

__all__ = [
    "get_probabilities",
    "get_int_labels",
    "calculate_accuracy",
    "calculate_weighted_accuracy",
    "get_dataset_len",
    "compute_pos_weights_h5",
    "compute_pos_weights_cloud",
    "compute_pos_weights",
    "compute_miou",
    "wrap_hist",
    "convert_str_values",
    "save_model",
    "load_model",
    "save2json",
    "load_json",
    "IouLoss",
    "DiceLoss",
    "ArcfaceFocalLoss",
    "FocalLoss",
    "DiscriminativeLoss",
    "calculate_l1_penalty_best_practice",
    "EarlyStopping",
]


def __getattr__(name):
    if name in {"Plotter", "classification_report"}:
        from .src import evaluation_plot_tools

        return getattr(evaluation_plot_tools, name)
    raise AttributeError(name)
