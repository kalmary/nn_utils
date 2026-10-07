import nn_utils
import subprocess
import sys
from pathlib import Path


def test_public_exports_cover_processing_utilities_without_plotting_imports():
    expected = {
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
    }

    assert set(nn_utils.__all__) == expected
    assert all(hasattr(nn_utils, name) for name in expected)
    assert "Plotter" not in nn_utils.__all__
    assert "classification_report" not in nn_utils.__all__


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


def test_json_helpers_import_without_torch_or_h5py():
    code = """
import builtins

original_import = builtins.__import__

def import_without_model_dependencies(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'h5py'}:
        raise ImportError(f'{name} is not available')
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_model_dependencies
from nn_utils import convert_str_values, load_json, save_to_json, wrap_hist
assert all(callable(item) for item in (convert_str_values, load_json, save_to_json, wrap_hist))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_tensor_helpers_import_without_h5py_or_plotting():
    code = """
import builtins

original_import = builtins.__import__

def import_without_offline_dependencies(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'h5py', 'matplotlib', 'pyvista', 'seaborn', 'sklearn'}:
        raise ImportError(f'{name} is not available')
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_offline_dependencies
from nn_utils import EarlyStopping, FocalLoss, calculate_accuracy
assert all(callable(item) for item in (EarlyStopping, FocalLoss, calculate_accuracy))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_plotting_names_remain_available_as_named_imports():
    from nn_utils import Plotter, classification_report

    assert callable(classification_report)
    assert callable(Plotter)
