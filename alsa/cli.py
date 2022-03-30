"""
Command-line interface of alsa.
"""
import json
from pathlib import Path
from typing import Optional

import typer

from alsa import crack_main, crack_train

APP = typer.Typer()

WORK_DIR_ARG = typer.Argument(..., exists=True, file_okay=False, dir_okay=True)


class PathEncoder(json.JSONEncoder):

    """
    Encoder for pathlib.Path objects.
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
    trace_width: float = typer.Option(0.1),
    cell_size: int = typer.Option(256),
    batch_size: int = typer.Option(64),
    old_weight_path: Optional[Path] = typer.Option(None),
    new_weight_path: Optional[Path] = typer.Option(None),
    training_plot_output: Optional[Path] = typer.Option(None),
    history_csv_path: Optional[Path] = typer.Option(None),
    quiet: bool = typer.Option(False),
    dry_run: bool = typer.Option(False),
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
        )
        typer.echo(f"Finished training in work directory: {work_dir.absolute()}")
    else:
        typer.echo("Finished dry run of training.")


@APP.command()
def predict(
    work_dir: Path = WORK_DIR_ARG,
    img_path: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    area_shp_file_path: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False
    ),
    unet_weights_path: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False
    ),
    new_shp_path: Path = typer.Option(..., file_okay=True, dir_okay=False),
    width: int = typer.Option(256),
    height: int = typer.Option(256),
    dry_run: bool = typer.Option(False),
):
    """
    Predict with trained model.
    """
    typer.echo(f"Work directory: {work_dir.absolute()}")
    typer.echo("Starting prediction.")

    if dry_run:
        return
    crack_main.crack_main(
        work_dir=work_dir,
        img_path=img_path,
        area_shp_file_path=area_shp_file_path,
        unet_weights_path=unet_weights_path,
        new_shp_path=new_shp_path,
        width=width,
        height=height,
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
