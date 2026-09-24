import nn_utils
import subprocess
import sys
from pathlib import Path


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


def test_basic_import_does_not_load_reporting_dependencies():
    code = (
        "import sys, nn_utils; "
        "assert not {'matplotlib', 'seaborn', 'sklearn'} & sys.modules.keys()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
