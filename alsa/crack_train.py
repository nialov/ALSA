"""
Methods for training model.
"""
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint

import alsa.geo_proc as gp
import alsa.image_proc as ip
from alsa.data import trainGenerator
from alsa.model import unet

# Directory structure is documented here as Paths
ORIG_IMG_DIR = Path("Training/Images/Originals")
SHP_BOUNDS_DIR = Path("Training/Shapefiles/Areas")
SHP_DIR = Path("Training/Shapefiles/Labels")
GEN_DIR = Path("Training/Images/Generated")
ORG_VAL_IMG_DIR = Path("Validation/Images/Originals")
VAL_IMG_DIR = Path("Validation/Images/Generated")
VAL_SHP_DIR = Path("Validation/Shapefiles/Labels")
VAL_BOUND_DIR = Path("Validation/Shapefiles/Areas")
SOURCE_SRC_DIR = GEN_DIR / "Source/src"
LABELS_LBL_DIR = GEN_DIR / "Labels/lbl"
PREDICTIONS_DIR = GEN_DIR / "Predictions"
SOURCE_V_SRC_V_DIR = VAL_IMG_DIR / "Source_v/src_v"
LABELS_V_LBL_V_DIR = VAL_IMG_DIR / "Labels_v/lbl_v"
PREDICTIONS_V_DIR = VAL_IMG_DIR / "Predictions"
WEIGHT_PATH = Path("unet_weights.hdf5")
PLOT_PATH = Path("training_plot.png")
HISTORY_CSV_PATH = Path("history.csv")


def match_images_to_labels_and_bounds(
    images: List[Path], trace_labels: List[Path], bounds: List[Path]
) -> List[Tuple[Path, Path, Path]]:
    """
    Match image files to labels (traces) and bounds (target areas).

    Used to match the images in training/validation folder to
    corresponding traces (labels) and areas (bounds).

    >>> images = [Path("kl5.png")]
    >>> trace_labels = [Path("kl5_traces.shp")]
    >>> bounds = [Path("kl5_area.shp")]
    >>> match_images_to_labels_and_bounds(images, trace_labels, bounds)
    [(PosixPath('kl5.png'), PosixPath('kl5_traces.shp'), PosixPath('kl5_area.shp'))]

    With faulty image name:

    >>> images = [Path("kl4.png")]
    >>> trace_labels = [Path("kl5_traces.shp")]
    >>> bounds = [Path("kl5_area.shp")]
    >>> match_images_to_labels_and_bounds(images, trace_labels, bounds)
    []

    Does not have to be ESRI Shapefiles.

    >>> images = [Path("kl5.png")]
    >>> trace_labels = [Path("kl5_traces.geojson")]
    >>> bounds = [Path("kl5_area.geojson")]
    >>> match_images_to_labels_and_bounds(images, trace_labels, bounds)
    [(PosixPath('kl5.png'), PosixPath('kl5_traces.geojson'), PosixPath('kl5_area.geojson'))]
    """
    target_list = []

    # Iterate over each image path
    for image in images:

        # Iterate over each traces path
        for label in trace_labels:
            # Iterate over each area path
            for bound in bounds:

                # Condition is as follows:
                # image name (stem) must be a subset of the names of traces and area
                if image.stem in label.name and image.stem in bound.name:
                    target_list.append((image, label, bound))
    return target_list


def preprocess_images(
    training_list: List[Tuple[Path, Path, Path]],
    source_src_dir: Path,
    labels_lbl_dir: Path,
    trace_width: float,
    cell_size: int,
):
    """
    Preprocess training or validation images.
    """
    for ind, (img, geo_file, geo_bounds) in enumerate(training_list):

        # Open image
        img = ip.open_image(img)

        # Read traces
        geo_data = gpd.read_file(geo_file)
        assert isinstance(geo_data, gpd.GeoDataFrame)

        # Read area (bounds)
        geo_area = gpd.read_file(geo_bounds)
        assert isinstance(geo_area, gpd.GeoDataFrame)

        # Resolve rectangular bounds from area
        min_x, min_y, max_x, max_y = geo_area.total_bounds

        # Determine the ratio used to transform trace_width
        # to the value of slack (buffer value around traces)
        real_to_pixel_ratio = gp.determine_real_to_pixel_ratio(
            image_shape=img.shape,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
        )
        slack = int(real_to_pixel_ratio * trace_width)

        # Create binary matrix from trace GeoDataFrame
        label_bin_img = gp.geo_dataframe_to_binmat(
            geo_data, img.shape, relative_geo_data=geo_area, slack=slack
        )

        # Divide original img (and binary images) into sub images of shape h, w
        # and save them
        w, h = (cell_size, cell_size)
        sub_imgs = ip.img_segmentation(img, width=w, height=h)
        sub_bin_imgs = ip.img_segmentation(label_bin_img, width=w, height=h)

        # Save images with a running suffix
        suf = 1
        for (im, b) in zip(sub_imgs, sub_bin_imgs):

            # Skip images that are too homogeneous (probably no traces)
            if np.quantile(im, 0.95) == 0 or np.quantile(im, 0.05) == 255:
                continue
            src_output_path = source_src_dir / f"{ind}_{suf}.png"
            ip.save_image(src_output_path, im)
            label_output_path = labels_lbl_dir / f"{ind}_{suf}.png"
            ip.save_image(label_output_path, b)
            suf += 1


def plot_training_process(
    epochs: int, fitted_model, training_plot_output: Optional[Path]
):
    """
    Plot training statistics and progress.
    """
    plt.style.use("ggplot")

    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("Epoch Number")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(
        np.arange(0, epochs),
        fitted_model.history["loss"],
        label="train_loss",
        color="green",
    )
    ax1.plot(
        np.arange(0, epochs),
        fitted_model.history["val_loss"],
        label="val_loss",
        color="red",
    )
    ax1.legend(loc="upper left")
    ax1.tick_params(axis="y", labelcolor=color)
    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    color2 = "tab:blue"
    # we already handled the x-label with ax1
    ax2.set_ylabel("Accuracy", color=color2)
    ax2.plot(
        np.arange(0, epochs),
        fitted_model.history["accuracy"],
        label="train_acc",
        color="black",
    )
    ax2.plot(
        np.arange(0, epochs),
        fitted_model.history["val_accuracy"],
        label="val_acc",
        color="blue",
    )
    ax2.tick_params(axis="y", labelcolor=color2)
    # Use tigh layout, otherwise the right y-label is slightly clipped
    fig.tight_layout()

    fig.savefig(training_plot_output, bbox_inches="tight")
    plt.close()


def create_generators(work_dir: Path, data_gen_args: Dict[str, Any], batch_size: int):
    """
    Create training and validation generators.
    """
    # training
    train_generator = trainGenerator(
        batch_size=batch_size,
        train_path=work_dir / GEN_DIR,
        image_folder="Source",
        mask_folder="Labels",
        aug_dict=data_gen_args,
        save_to_dir=None,
    )

    # validation
    val_generator = trainGenerator(
        batch_size=batch_size,
        train_path=work_dir / VAL_IMG_DIR,
        image_folder="Source_v",
        mask_folder="Labels_v",
        aug_dict=data_gen_args,
        save_to_dir=None,
    )
    return train_generator, val_generator


def collect_targets(work_dir: Path, spatial_file_extension: str = "shp"):
    """
    Collect and associate training and validation images.

    Associates images to traces (labels) and areas (bounds)
    based on filenames in training and validation directories.
    """
    # Training paths
    orig_img_dir = work_dir / ORIG_IMG_DIR
    orig_shp_dir = work_dir / SHP_DIR
    shp_bounds_dir = work_dir / SHP_BOUNDS_DIR

    # Validation paths
    org_val_img_dir = work_dir / ORG_VAL_IMG_DIR
    val_shp_dir = work_dir / VAL_SHP_DIR
    val_bound_dir = work_dir / VAL_BOUND_DIR

    target_dict = dict()

    # Conduct same process for both sets of paths (training and validation)
    for img_dir, shp_dir, bounds_dir, target in zip(
        (orig_img_dir, org_val_img_dir),
        (orig_shp_dir, val_shp_dir),
        (shp_bounds_dir, val_bound_dir),
        ("training", "validation"),
    ):

        images = list(img_dir.glob("*.png")) if img_dir.exists() else []
        trace_labels = (
            list(shp_dir.glob(f"*.{spatial_file_extension}"))
            if shp_dir.exists()
            else []
        )
        bounds = (
            list(bounds_dir.glob(f"*.{spatial_file_extension}"))
            if bounds_dir.exists()
            else []
        )

        # Match images to traces and areas
        target_list = match_images_to_labels_and_bounds(
            images=images,
            trace_labels=trace_labels,
            bounds=bounds,
        )
        target_dict[target] = target_list

    # Return two lists, one for training, other for validation
    return target_dict["training"], target_dict["validation"]


def preprocess_training_and_validation(
    work_dir: Path,
    training_list: list,
    validation_list: list,
    trace_width: float,
    cell_size: int,
):
    """
    Create sub-images of training and validation data.
    """
    preprocess_images(
        training_list=training_list,
        source_src_dir=work_dir / SOURCE_SRC_DIR,
        labels_lbl_dir=work_dir / LABELS_LBL_DIR,
        trace_width=trace_width,
        cell_size=cell_size,
    )
    preprocess_images(
        training_list=validation_list,
        source_src_dir=work_dir / SOURCE_V_SRC_V_DIR,
        labels_lbl_dir=work_dir / LABELS_V_LBL_V_DIR,
        trace_width=trace_width,
        cell_size=cell_size,
    )


def train_main(
    epochs: int,
    validation_steps: int,
    steps_per_epoch: int,
    trace_width: float,
    cell_size: int = 256,
    batch_size: int = 64,
    work_dir: Path = Path("."),
    old_weight_path: Optional[Path] = None,
    new_weight_path: Optional[Path] = None,
    training_plot_output: Optional[Path] = None,
    history_csv_path: Optional[Path] = None,
    spatial_file_extension: str = "shp",
):
    """
    Train model.

    Continues training the model with weights in old_weight_path. If None,
    starts from scratch and saves it to working directory. New weights will be
    saved to a file in new_weight_path. If None, it will instead overwrite the
    old one.
    """
    # Resolve full path for safety
    work_dir = work_dir.resolve()

    # Set default weights path if none given
    if new_weight_path is None:
        new_weight_path = work_dir / WEIGHT_PATH

    # Create model from old weights, if they were given
    model = unet(old_weight_path)

    # Associate training and validation images with trace and area data.
    training_list, validation_list = collect_targets(
        work_dir=work_dir, spatial_file_extension=spatial_file_extension
    )

    # Check that training and validation targets are found
    # and correctly associated
    for target_list, target in zip(
        (training_list, validation_list), ("training", "validation")
    ):
        if len(target_list) == 0:
            raise FileNotFoundError(
                f"Expected to find .png-.{spatial_file_extension} {target} data combinations."
            )

    # Create sub-images of both training and validation data in correct directories.
    preprocess_training_and_validation(
        work_dir=work_dir,
        training_list=training_list,
        validation_list=validation_list,
        trace_width=trace_width,
        cell_size=cell_size,
    )

    # Set up the training generator's image altering parameters
    data_gen_args = dict(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    train_generator, val_generator = create_generators(
        work_dir=work_dir,
        data_gen_args=data_gen_args,
        batch_size=batch_size,
    )

    # Monitoring validation loss instead of training loss
    model_checkpoint = ModelCheckpoint(
        new_weight_path, monitor="val_loss", mode="min", verbose=1, save_best_only=True
    )

    # Fit the model
    fitted_model = model.fit(
        train_generator,
        validation_data=val_generator,
        validation_steps=validation_steps,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[model_checkpoint],
    )

    # Convert the history.history dict to a pandas DataFrame:
    hist_df_ = pd.DataFrame(fitted_model.history)

    # Save to csv:
    if history_csv_path is None:
        history_csv_path = work_dir / HISTORY_CSV_PATH
    hist_df_.to_csv(history_csv_path)

    # model.evaluate(val_generator)

    # Plot training progress
    if training_plot_output is None:
        training_plot_output = work_dir / "training_plot.png"
    plot_training_process(
        epochs=epochs,
        fitted_model=fitted_model,
        training_plot_output=training_plot_output,
    )


def train_directory_setup(work_dir: Path = Path(".")):
    """
    Set up training and validation directories.

    Tries to find necessary files for the training. If fails,
    it will create them to the working directory.
    """
    training_dirs = [
        ORIG_IMG_DIR,
        SHP_BOUNDS_DIR,
        SHP_DIR,
        GEN_DIR,
        ORG_VAL_IMG_DIR,
        VAL_IMG_DIR,
        VAL_SHP_DIR,
        VAL_BOUND_DIR,
        SOURCE_SRC_DIR,
        LABELS_LBL_DIR,
        PREDICTIONS_DIR,
        SOURCE_V_SRC_V_DIR,
        LABELS_V_LBL_V_DIR,
        PREDICTIONS_V_DIR,
    ]
    paths = [work_dir / train_dir for train_dir in training_dirs]
    for dir_path in paths:
        if (
            dir_path.stem == "Generated"
            and "Training" in str(dir_path)
            and dir_path.exists()
        ):
            # Remove directory with generated training data
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
