from importlib import import_module


_EXPORT_MODULES = {
    "get_probabilities": ".src.accuracy_metrics",
    "get_int_labels": ".src.accuracy_metrics",
    "calculate_accuracy": ".src.accuracy_metrics",
    "calculate_weighted_accuracy": ".src.accuracy_metrics",
    "get_dataset_len": ".src.accuracy_metrics",
    "compute_pos_weights_h5": ".src.accuracy_metrics",
    "compute_pos_weights_cloud": ".src.accuracy_metrics",
    "compute_pos_weights": ".src.accuracy_metrics",
    "compute_miou": ".src.accuracy_metrics",
    "wrap_hist": ".src.file_handling",
    "convert_str_values": ".src.file_handling",
    "save_model": ".src.file_handling",
    "load_model": ".src.file_handling",
    "save_to_json": ".src.file_handling",
    "load_json": ".src.file_handling",
    "IouLoss": ".src.loss_functions",
    "DiceLoss": ".src.loss_functions",
    "ArcfaceFocalLoss": ".src.loss_functions",
    "FocalLoss": ".src.loss_functions",
    "DiscriminativeLoss": ".src.loss_functions",
    "calculate_l1_penalty_best_practice": ".src.loss_functions",
    "EarlyStopping": ".src.training_callbacks",
    "Plotter": ".src.evaluation_plot_tools",
    "classification_report": ".src.evaluation_plot_tools",
}

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
    "save_to_json",
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
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
