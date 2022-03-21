"""
Parameters for tests.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from shutil import copy

KL5_TEST_DATA_DIR = Path("tests/sample_data/kl5_test_data")

KL5_TEST_AREA_DIR = KL5_TEST_DATA_DIR / "area/"
KL5_TEST_TRACES_DIR = KL5_TEST_DATA_DIR / "traces/"
KL5_TEST_IMAGE = KL5_TEST_DATA_DIR / "kl5_subsample_circle.png"
KL5_TEST_WEIGHTS = KL5_TEST_DATA_DIR / "unet_weights.hdf5"


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


def copy_files_from_dir_to_dir(input_dir: Path, output_dir: Path):
    """
    Copy all files from input_dir to output_dir.
    """
    for path in input_dir.iterdir():
        if path.is_file():
            copy(path, output_dir)


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