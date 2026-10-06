import numpy as np
import pytest

from nn_utils import compute_pos_weights_cloud


def test_cloud_weights_reject_missing_or_empty_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        compute_pos_weights_cloud(tmp_path / "missing", num_classes=2)

    with pytest.raises(FileNotFoundError, match="No .npy point-cloud tiles"):
        compute_pos_weights_cloud(tmp_path, num_classes=2)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (np.zeros((2, 4)), r"Expected \(N, 5\) data"),
        (np.empty((0, 5)), "contains no points"),
        (
            np.array([[0.0, 0.0, 0.0, 0.0, np.nan]]),
            "Non-finite labels",
        ),
        (
            np.array([[0.0, 0.0, 0.0, 0.0, 0.5]]),
            "Non-integer labels",
        ),
        (
            np.array([[0.0, 0.0, 0.0, 0.0, 2.0]]),
            "Label outside configured classes",
        ),
    ],
)
def test_cloud_weights_validate_tile_contents(tmp_path, data, message):
    np.save(tmp_path / "tile.npy", data)

    with pytest.raises(ValueError, match=message):
        compute_pos_weights_cloud(tmp_path, num_classes=2)
