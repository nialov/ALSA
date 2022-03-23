"""
Test crack_train.py
"""
import os
from pathlib import Path
from shutil import copy

import pytest

import tests
from alsa import crack_train


@pytest.fixture
def setup_train_test(tmp_path: Path):
    """
    Setup training data for test(s).
    """
    assert len(list(tmp_path.iterdir())) == 0

    # Setup directories
    crack_train.train_directory_setup(tmp_path)

    # Copy test data to training and validation directories
    tests.copy_files_from_dir_to_dir(
        tests.KL5_TEST_AREA_DIR, tmp_path / crack_train.SHP_BOUNDS_DIR
    )
    tests.copy_files_from_dir_to_dir(
        tests.KL5_TEST_TRACES_DIR, tmp_path / crack_train.SHP_DIR
    )
    copy(tests.KL5_TEST_IMAGE, tmp_path / crack_train.ORIG_IMG_DIR)
    tests.copy_files_from_dir_to_dir(
        tests.KL5_TEST_AREA_DIR, tmp_path / crack_train.VAL_BOUND_DIR
    )
    tests.copy_files_from_dir_to_dir(
        tests.KL5_TEST_TRACES_DIR, tmp_path / crack_train.VAL_SHP_DIR
    )
    copy(tests.KL5_TEST_IMAGE, tmp_path / crack_train.ORG_VAL_IMG_DIR)
    return tmp_path


def test_setup_training(setup_train_test):
    """
    Test setup_training.
    """
    tmp_path = setup_train_test

    training_list, validation_list = crack_train.setup_training(tmp_path)

    assert len(training_list) > 0

    # Same data used in this case
    assert len(training_list) == len(validation_list)

    return training_list, validation_list


def test_preprocess_images(setup_train_test):
    """
    Test preprocess_images.
    """
    tmp_path = setup_train_test
    assert isinstance(tmp_path, Path)
    training_list, _ = test_setup_training(tmp_path)

    source_src_dir = tmp_path / crack_train.SOURCE_SRC_DIR
    labels_lbl_dir = tmp_path / crack_train.LABELS_LBL_DIR

    crack_train.preprocess_images(
        training_list=training_list,
        source_src_dir=source_src_dir,
        labels_lbl_dir=labels_lbl_dir,
    )

    assert len(list(source_src_dir.glob("*.png"))) > 0
    assert len(list(labels_lbl_dir.glob("*.png"))) > 0


@pytest.mark.skipif(
    os.environ.get("CI") is not None, reason="Tensorflow crashes on Github Actions."
)
def test_train(setup_train_test):
    """
    Test model training.
    """
    tmp_path = setup_train_test

    # Conduct training
    crack_train.train_main(
        epochs=1, validation_steps=1, steps_per_epoch=1, work_dir=tmp_path
    )

    # Test that training has created expected files
    assert (tmp_path / crack_train.WEIGHT_PATH).exists()
    assert (tmp_path / crack_train.HISTORY_CSV_PATH).exists()

    # Check generated directories for both training and validation
    for gen_path in (
        crack_train.SOURCE_SRC_DIR,
        crack_train.LABELS_LBL_DIR,
        crack_train.SOURCE_V_SRC_V_DIR,
        crack_train.LABELS_V_LBL_V_DIR,
    ):
        true_path = tmp_path / gen_path
        assert len(list(true_path.iterdir())) > 0


@pytest.mark.parametrize(
    "images, trace_labels, bounds, assumed_result",
    tests.match_images_to_labels_and_bounds_params(),
)
def test_match_images_to_labels_and_bounds(
    images, trace_labels, bounds, assumed_result
):
    """
    Test match_images_to_labels_and_bounds.
    """
    result = crack_train.match_images_to_labels_and_bounds(images, trace_labels, bounds)

    assert len(result) == len(assumed_result)
    for path_tuple, correct_path_tuple in zip(result, assumed_result):
        assert path_tuple == correct_path_tuple
