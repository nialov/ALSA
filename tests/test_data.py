"""
Tests for alsa.data.
"""

import pytest

import tests
from alsa import data


@pytest.mark.parametrize(
    "array,n_mats_per_row,n_mats_per_col", tests.test_save_result_params()
)
def test_save_result(array, n_mats_per_row, n_mats_per_col, tmp_path):
    """
    Test save_result.
    """
    data.save_result(
        save_path=tmp_path,
        array=array,
        n_mats_per_col=n_mats_per_col,
        n_mats_per_row=n_mats_per_row,
    )

    mosaic_path = tmp_path / data.MOSAIC_PREDICT_PATH
    assert mosaic_path.exists()

    assert len(list(tmp_path.glob("*.png"))) > 1
