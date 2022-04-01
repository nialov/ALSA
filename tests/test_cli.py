"""
Tests for alsa.cli.
"""

import os
from pathlib import Path
from traceback import print_tb

import geopandas as gpd
import pytest
from click.testing import Result
from typer.testing import CliRunner

import tests
from alsa import cli, crack_train
from tests.test_crack_train import add_kl5_training_data

RUNNER = CliRunner()


def click_error_print(result: Result):
    """
    Print click result traceback.
    """
    if result.exit_code == 0:
        return
    assert result.exc_info is not None
    _, _, tb = result.exc_info
    # print(err_class, err)
    print_tb(tb)
    print(result.output)
    raise Exception(result.exception)


def _test_cli_train(
    tmp_path,
    epochs,
    validation_steps,
    steps_per_epoch,
    trace_width,
    cell_size,
    batch_size,
    quiet,
    dry_run,
):
    """
    Test command-line interface to training.
    """
    add_kl5_training_data(tmp_path=tmp_path, rename_count=1)

    args = [
        "train",
        str(tmp_path),
        f"--epochs={epochs}",
        f"--validation-steps={validation_steps}",
        f"--steps-per-epoch={steps_per_epoch}",
        f"--trace-width={trace_width}",
        f"--cell-size={cell_size}",
        f"--batch-size={batch_size}",
    ]
    if quiet:
        args.append("--quiet")
    if dry_run:
        args.append("--dry-run")
    # Conduct training from cli
    result = RUNNER.invoke(cli.APP, args)
    click_error_print(result)
    if dry_run:
        assert not (tmp_path / crack_train.WEIGHT_PATH).exists()
        return

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


@pytest.mark.skipif(
    os.environ.get("CI") is not None, reason="Tensorflow crashes on Github Actions."
)
@pytest.mark.parametrize(
    ",".join(
        [
            "epochs",
            "validation_steps",
            "steps_per_epoch",
            "trace_width",
            "cell_size",
            "batch_size",
            "quiet",
            "dry_run",
        ]
    ),
    [
        (
            1,
            1,
            1,
            0.1,
            512,
            1,
            False,
            False,
        ),
        (
            1,
            1,
            1,
            0.1,
            512,
            1,
            False,
            True,
        ),
        (
            1,
            1,
            1,
            0.1,
            512,
            1,
            True,
            True,
        ),
    ],
)
def test_cli_train(
    setup_train_test,
    epochs,
    validation_steps,
    steps_per_epoch,
    trace_width,
    cell_size,
    batch_size,
    quiet,
    dry_run,
):
    """
    Test command-line interface to training.
    """
    tmp_path = setup_train_test
    _test_cli_train(
        tmp_path=tmp_path,
        epochs=epochs,
        validation_steps=validation_steps,
        steps_per_epoch=steps_per_epoch,
        trace_width=trace_width,
        cell_size=cell_size,
        batch_size=batch_size,
        quiet=quiet,
        dry_run=dry_run,
    )


@pytest.mark.parametrize("quiet", [True, False])
def test_cli_train_interface(setup_train_test, quiet):
    """
    Test cli.predict interface but not actual training.

    Should be runnable on Github Actions.
    """
    tmp_path = setup_train_test
    _test_cli_train(
        tmp_path=tmp_path,
        epochs=1,
        validation_steps=1,
        steps_per_epoch=1,
        trace_width=0.1,
        cell_size=512,
        batch_size=1,
        quiet=quiet,
        dry_run=True,
    )


@pytest.mark.parametrize("width,height", [(100, 100), (256, 256)])
def test_cli_predict_interface(tmp_path, width, height):
    """
    Test predict cli interface but not actual prediction.

    Should be runnable on Github Actions.
    """
    img_path = tests.KL5_TEST_IMAGE
    area_shp_file_path = list(tests.KL5_TEST_AREA_DIR.glob("*.shp"))[0]
    unet_weights_path = tmp_path / "weights.hdf5"
    unet_weights_path.touch()
    new_shp_path = tmp_path / "predicted_traces.shp"
    args = [
        "predict",
        str(tmp_path),
        f"--img-path={img_path}",
        f"--area-file-path={area_shp_file_path}",
        f"--unet-weights-path={unet_weights_path}",
        f"--predicted-output-path={new_shp_path}",
        f"--width={width}",
        f"--height={height}",
        "--dry-run",
    ]
    # Call predict cli interface with dry-run enabled
    result = RUNNER.invoke(cli.APP, args)

    click_error_print(result)


@pytest.mark.skipif(
    os.environ.get("CI") is not None, reason="Tensorflow crashes on Github Actions."
)
def test_train_and_predict(setup_train_test):
    """
    Test training and prediction.
    """
    tmp_path = setup_train_test

    # Training
    cell_size = 512
    _test_cli_train(
        tmp_path=tmp_path,
        epochs=1,
        validation_steps=1,
        steps_per_epoch=1,
        trace_width=0.1,
        cell_size=cell_size,
        batch_size=1,
        quiet=False,
        dry_run=False,
    )

    # Prediction
    img_path = tests.KL5_TEST_IMAGE
    area_shp_file_path = list(tests.KL5_TEST_AREA_DIR.glob("*.shp"))[0]
    unet_weights_path = tmp_path / crack_train.WEIGHT_PATH

    assert unet_weights_path.exists()

    new_shp_path = tmp_path / "predicted_traces.shp"

    assert not new_shp_path.exists()
    args = [
        "predict",
        str(tmp_path),
        f"--img-path={img_path}",
        f"--area-file-path={area_shp_file_path}",
        f"--unet-weights-path={unet_weights_path}",
        f"--predicted-output-path={new_shp_path}",
        f"--width={cell_size}",
        f"--height={cell_size}",
    ]

    # Call predict cli interface
    result = RUNNER.invoke(cli.APP, args)

    click_error_print(result)

    assert new_shp_path.exists()

    gdf = gpd.read_file(new_shp_path)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.shape[0] > 0


@pytest.mark.parametrize("setup_dirs", [True, False])
def test_cli_check(tmp_path: Path, setup_dirs: bool):
    """
    Test cli.check.
    """
    args = [
        "check",
        str(tmp_path),
    ]

    if setup_dirs:
        args.append("--setup-dirs")

    orig_img_dir = tmp_path / crack_train.ORIG_IMG_DIR
    val_img_dir = tmp_path / crack_train.VAL_IMG_DIR
    assert not orig_img_dir.exists()
    assert not val_img_dir.exists()

    # Call check cli interface
    result = RUNNER.invoke(cli.APP, args)

    click_error_print(result)

    if setup_dirs:
        assert orig_img_dir.exists() and orig_img_dir.is_dir()
        assert val_img_dir.exists() and val_img_dir.is_dir()
