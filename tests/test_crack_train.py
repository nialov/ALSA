"""
Test crack_train.py
"""
import os
from pathlib import Path
from shutil import copy

import pytest

import tests
from alsa import crack_train


def add_kl5_training_data(tmp_path: Path, rename_count: int = 1):
    """
    Add kl5 sample data to correct directories.

    Using the rename_count argument you can rename the sample data
    a specified number of times to e.g.
    test training with multiple images and associated labels and bounds.
    Same kl5 data will however be used.
    """
    for i in range(rename_count):
        rename = ("kl5", f"kl{5+i}")
        for area_dir in (crack_train.SHP_BOUNDS_DIR, crack_train.VAL_BOUND_DIR):
            tests.copy_files_from_dir_to_dir(
                tests.KL5_TEST_AREA_DIR, tmp_path / area_dir, rename=rename
            )

        for traces_dir in (crack_train.SHP_DIR, crack_train.VAL_SHP_DIR):
            # Copy test data to training and validation directories
            tests.copy_files_from_dir_to_dir(
                tests.KL5_TEST_TRACES_DIR, tmp_path / traces_dir, rename=rename
            )

        output_img_name = tests.KL5_TEST_IMAGE.name
        if len(rename) == 2:
            output_img_name = output_img_name.replace(*rename)

        for img_dir in (crack_train.ORIG_IMG_DIR, crack_train.ORG_VAL_IMG_DIR):
            copy(tests.KL5_TEST_IMAGE, tmp_path / img_dir / output_img_name)

    assert len(list(tmp_path.rglob("*.png"))) == rename_count * 2


def _test_setup_training(tmp_path: Path, rename_count: int):
    add_kl5_training_data(tmp_path=tmp_path, rename_count=rename_count)

    training_list, validation_list = crack_train.collect_targets(
        tmp_path,
    )

    assert len(training_list) > 0

    # Same data used in this case
    assert len(training_list) == len(validation_list)

    return training_list, validation_list


@pytest.mark.parametrize("rename_count", [2, 3])
def test_setup_training(setup_train_test, rename_count):
    """
    Test setup_training.
    """
    tmp_path = setup_train_test
    _test_setup_training(tmp_path=tmp_path, rename_count=rename_count)


def test_preprocess_images(setup_train_test):
    """
    Test preprocess_images.
    """
    tmp_path = setup_train_test
    add_kl5_training_data(tmp_path=tmp_path)
    assert isinstance(tmp_path, Path)
    trace_width = 0.1
    training_list, _ = _test_setup_training(tmp_path, rename_count=1)

    source_src_dir = tmp_path / crack_train.SOURCE_SRC_DIR
    labels_lbl_dir = tmp_path / crack_train.LABELS_LBL_DIR

    crack_train.preprocess_images(
        training_list=training_list,
        source_src_dir=source_src_dir,
        labels_lbl_dir=labels_lbl_dir,
        cell_size=512,
        trace_width=trace_width,
    )

    assert len(list(source_src_dir.glob("*.png"))) > 0
    assert len(list(labels_lbl_dir.glob("*.png"))) > 0


@pytest.mark.skipif(
    os.environ.get("CI") is not None, reason="Tensorflow crashes on Github Actions."
)
@pytest.mark.parametrize("rename_count, batch_size", [(1, 8), (2, 4)])
def test_train(setup_train_test, rename_count, batch_size):
    """
    Test model training.
    """
    tmp_path = setup_train_test
    add_kl5_training_data(tmp_path=tmp_path, rename_count=rename_count)

    # Conduct training
    crack_train.train_main(
        epochs=1,
        validation_steps=1,
        steps_per_epoch=1,
        work_dir=tmp_path,
        trace_width=0.1,
        batch_size=batch_size,
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
