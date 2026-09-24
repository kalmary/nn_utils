import nn_utils


def test_public_exports_cover_processing_utilities_without_plotting_imports():
    expected = {
        "get_Probabilities",
        "get_intLabels",
        "calculate_accuracy",
        "calculate_weighted_accuracy",
        "get_dataset_len",
        "compute_pos_weights_h5",
        "compute_pos_weights_cloud",
        "compute_pos_weights",
        "compute_mIoU",
        "wrap_hist",
        "convert_str_values",
        "save_model",
        "load_model",
        "save2json",
        "load_json",
        "IoULoss",
        "DiceLoss",
        "ArcFaceFocalLoss",
        "FocalLoss",
        "DiscriminativeLoss",
        "calculate_l1_penalty_best_practice",
        "EarlyStopping",
    }

    assert set(nn_utils.__all__) == expected
    assert all(hasattr(nn_utils, name) for name in expected)
    assert "Plotter" not in nn_utils.__all__
    assert "ClassificationReport" not in nn_utils.__all__
