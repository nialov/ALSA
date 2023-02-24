"""
Command-line interface of alsa.
"""
import json
from pathlib import Path
from typing import Optional

import typer
from fiona import supported_drivers

from alsa import crack_main, crack_train

APP = typer.Typer()

WORK_DIR_ARG = typer.Argument(
    ...,
    exists=True,
    file_okay=False,
    dir_okay=True,
    help="Working directory. Note that conflicting files are overridden without warning.",
)


class PathEncoder(json.JSONEncoder):

    """
    Encoder for pathlib.Path objects.

    Used to print paths as json objects.
    """

    def default(self, obj):
        """
        Override default method.
        """
        if isinstance(obj, Path):
            return str(obj)

        # Use default class method for other types
        return json.JSONEncoder.default(self, obj)


def report_target_lists(work_dir: Path):
    """
    Report training and validation targets.
    """
    # Gathered here just for reporting to the user.
    # Are gathered again in `train_main` function.
    training_list, validation_list = crack_train.collect_targets(work_dir=work_dir)

    for target_list, name in zip(
        (training_list, validation_list), ("Training", "Validation")
    ):
        # Convert to json for more human-readable printing
        list_json = json.dumps(target_list, indent=1, cls=PathEncoder)
        typer.echo(f"{name} data pairs: {list_json}")


@APP.command()
def train(
    work_dir: Path = WORK_DIR_ARG,
    epochs: int = typer.Option(100),
    validation_steps: int = typer.Option(10),
    steps_per_epoch: int = typer.Option(10),
    trace_width: float = typer.Option(
        0.01, help="Width of traces used in training in coordinate system units."
    ),
    cell_size: int = typer.Option(256, help="Size of sub-image cell in training."),
    batch_size: int = typer.Option(64, help="trainGenerator batch size."),
    old_weight_path: Optional[Path] = typer.Option(
        None,
        help=f"Defaults to <work_dir>/{crack_train.WEIGHT_PATH}",
    ),
    new_weight_path: Optional[Path] = typer.Option(
        None,
        help=f"Defaults to <work_dir>/{crack_train.WEIGHT_PATH}",
    ),
    training_plot_output: Optional[Path] = typer.Option(
        None,
        help=f"Defaults to <work_dir>/{crack_train.PLOT_PATH}",
    ),
    history_csv_path: Optional[Path] = typer.Option(
        None,
        help=f"Defaults to <work_dir>/{crack_train.HISTORY_CSV_PATH}",
    ),
    quiet: bool = typer.Option(False, help="Control verbosity (prints to stdout)."),
    dry_run: bool = typer.Option(False, help="Do not train."),
    spatial_file_extension: str = typer.Option(
        "shp", help="Give the extension of the input spatial filetype."
    ),
):
    """
    Train model.
    """
    # typer.echo is an alternative to print.
    typer.echo(f"Work directory: {work_dir.absolute()}")
    typer.echo("Setting up training and validation directories.")

    # Setup training and validation directories i.e. create them as needed
    if not dry_run:
        crack_train.train_directory_setup(work_dir=work_dir)

    if not quiet:
        report_target_lists(work_dir=work_dir)

    # Report input parameters as json
    relevant_input_params = json.dumps(
        dict(
            epochs=epochs,
            validation_steps=validation_steps,
            steps_per_epoch=steps_per_epoch,
            trace_width=trace_width,
            cell_size=cell_size,
            batch_size=batch_size,
        ),
        indent=1,
    )
    if not quiet:
        typer.echo(relevant_input_params)

    # Start training
    typer.echo("Starting training.")

    if not dry_run:
        crack_train.train_main(
            work_dir=work_dir,
            epochs=epochs,
            validation_steps=validation_steps,
            steps_per_epoch=steps_per_epoch,
            trace_width=trace_width,
            cell_size=cell_size,
            batch_size=batch_size,
            old_weight_path=old_weight_path,
            new_weight_path=new_weight_path,
            training_plot_output=training_plot_output,
            history_csv_path=history_csv_path,
            spatial_file_extension=spatial_file_extension,
        )
        typer.echo(f"Finished training in work directory: {work_dir.absolute()}")
    else:
        typer.echo("Finished dry run of training.")


def check_driver(driver: str):
    """
    Check that driver is supported by fiona.
    """
    error = "\n".join(
        [
            "Expected driver to be writable by fiona/geopandas.",
            "See supported drivers and capabilities:",
            str(json.dumps(supported_drivers, indent=1)),
            "r=read, w=write, a=append",
            "Write capability is required.",
        ]
    )
    if driver not in supported_drivers or "w" not in supported_drivers[driver]:
        raise typer.BadParameter(error)
    return driver


@APP.command()
def predict(
    work_dir: Path = WORK_DIR_ARG,
    img_path: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to png-image to predict on.",
    ),
    area_file_path: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to area bounds.",
    ),
    unet_weights_path: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to trained hdf5 weights.",
    ),
    predicted_output_path: Path = typer.Option(
        ..., file_okay=True, dir_okay=False, help="Path to predicted traces output."
    ),
    width: int = typer.Option(256, help="Height of sub-images."),
    height: int = typer.Option(256, help="Height of sub-images."),
    dry_run: bool = typer.Option(False, help="Do not run prediction."),
    override_ridge_config_path: Optional[Path] = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help=(
            "The ridge detections config can be overridden by passing a"
            " json file with wanted configuration."
            " By default will use <work_dir>/ridge_config.json if found."
        ),
    ),
    quiet: bool = typer.Option(False, help="Control verbosity (prints to stdout)."),
    driver: str = typer.Option(
        "ESRI Shapefile",
        callback=check_driver,
        help="Choose spatial driver for output file.",
    ),
):
    """
    Predict with trained model.
    """
    typer.echo(f"Work directory: {work_dir.absolute()}")
    typer.echo("Starting prediction.")

    if dry_run:
        # These are useful for simple tests of the command-line interface
        return

    crack_main.crack_main(
        work_dir=work_dir,
        img_path=img_path,
        area_file_path=area_file_path,
        unet_weights_path=unet_weights_path,
        predicted_output_path=predicted_output_path,
        width=width,
        height=height,
        # The ridge detections config can be overridden
        # by passing a json file with wanted configuration.
        override_ridge_config=override_ridge_config_path,
        verbose=not quiet,
        driver=driver,
    )


@APP.command()
def check(
    work_dir: Path = WORK_DIR_ARG,
    setup_dirs: bool = typer.Option(False, help="Create target directories."),
):
    """
    Check training and validation targets and report to user.
    """
    report_target_lists(work_dir=work_dir)
    if setup_dirs:
        crack_train.train_directory_setup(work_dir=work_dir)
