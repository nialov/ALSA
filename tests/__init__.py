"""
Parameters for tests.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from shutil import copy
from typing import Tuple

import numpy as np

KL5_TEST_DATA_DIR = Path("tests/sample_data/kl5_test_data")

KL5_TEST_AREA_DIR = KL5_TEST_DATA_DIR / "area/"
KL5_TEST_TRACES_DIR = KL5_TEST_DATA_DIR / "traces/"
KL5_TEST_IMAGE = KL5_TEST_DATA_DIR / "kl5_subsample.png"
KL5_TEST_IMAGE_16BIT = KL5_TEST_DATA_DIR / "kl5_subsample_16bit.png"
KL5_TEST_WEIGHTS = KL5_TEST_DATA_DIR / "unet_weights.hdf5"
SAMPLE_RIDGE_CONFIG_PATH = Path("tests/sample_data/ridge_config.json")


@contextmanager
def change_dir(path: Path):
    """
    Change directory to path during with-context.
    """
    original_dir = Path(".").absolute()
    try:
        yield os.chdir(path)
    finally:
        os.chdir(original_dir)


def copy_files_from_dir_to_dir(
    input_dir: Path, output_dir: Path, rename: Tuple[str, str]
):
    """
    Copy all files from input_dir to output_dir.
    """
    for path in input_dir.iterdir():
        if path.is_file():
            output_name = path.name
            if len(rename) == 2:
                output_name = output_name.replace(*rename)
            output_path = output_dir / output_name
            copy(path, output_path)


def match_images_to_labels_and_bounds_params():
    """
    Params for match_images_to_labels_and_bounds.
    """
    kl5_img, kl5_traces, kl5_area = (
        Path("kl5.png"),
        Path("kl5_traces.shp"),
        Path("kl5_area.shp"),
    )
    kl5_traces_wrong = Path("kl6_traces.shp")
    correct_params = (
        [kl5_img],
        [kl5_traces],
        [kl5_area],
        [(kl5_img, kl5_traces, kl5_area)],
    )
    incorrect_params = (
        [kl5_img],
        [kl5_traces_wrong],
        [kl5_area],
        [],
    )
    return [
        correct_params,
        incorrect_params,
    ]


def test_resolve_ridge_config_overrides_params():
    return [
        None,
        SAMPLE_RIDGE_CONFIG_PATH,
    ]


def test_save_result_params():
    arr = np.array(
        [
            [(0.5, 0.5, 0.1), (0.5, 0.5, 0.5)],
            [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
        ]
    )
    return [
        (
            np.array(
                [
                    arr,
                    arr,
                    arr,
                    arr,
                ]
            ),
            2,
            2,
        ),
    ]
