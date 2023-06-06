#!/usr/bin/env python3

import argparse
import re
import shutil
import subprocess
from enum import Enum, unique
from pathlib import Path
from typing import NamedTuple

DOWNLOADS_DIR_NAME = "downloads"

TRACES_URL = "https://zenodo.org/record/7077846/files/loviisa_manual_traces_and_areas.zip?download=1"
TRACES_ZIP_NAME = "U-Net_Traces.zip"
TRACES_DIR_NAME = Path(TRACES_ZIP_NAME).stem
TRACES_DIR_TRACES = "data-exported-ESRI-Shapefile/loviisa/traces/20m/"
TRACES_DIR_AREAS = "data-exported-ESRI-Shapefile/loviisa/area/20m/"

IMAGES_URL = "https://zenodo.org/record/7974402/files/Loviisa_orthomosaics_for_automation.zip?download=1"
IMAGES_ZIP_NAME = "Loviisa_orthomosaics_for_automation.zip"
IMAGES_DIR_NAME = Path(IMAGES_ZIP_NAME).stem

MODEL_URL = (
    "https://zenodo.org/record/7077620/files/unet_weights_2022_a.hdf5?download=1"
)

# From alsa/crack_train.py
TRAINING_IMAGES = Path("Training/Images/Originals")
TRAINING_AREAS = Path("Training/Shapefiles/Areas")
TRAINING_TRACES = Path("Training/Shapefiles/Labels")

VALIDATION_IMAGES = Path("Validation/Images/Originals")
VALIDATION_AREAS = Path("Validation/Shapefiles/Areas")
VALIDATION_TRACES = Path("Validation/Shapefiles/Labels")

PREDICTION_IMAGES = Path("prediction/Images")
PREDICTION_AREAS = Path("prediction/Shapefiles/Areas")
PREDICTION_TRACES = Path("prediction/Shapefiles/Labels")


@unique
class DataType(Enum):
    TRAINING = "training"
    VALIDATION = "Validation"
    PREDICTION = "prediction"


class Data(NamedTuple):
    name: str
    traces: str
    area: str
    image: str
    data_type: DataType


TRACE_AREA_IMAGE_DATA_PAIRS = [
    Data(
        name="kb11",
        traces="KB11_tulkinta.shp",
        area="kb11_area.shp",
        image="KB11.png",
        data_type=DataType.VALIDATION,
    ),
    Data(
        name="kb2",
        traces="KB2_tulkinta_clip.shp",
        area="kb2_area.shp",
        image="KB2.png",
        data_type=DataType.TRAINING,
    ),
    Data(
        name="kb3",
        traces="KB3_tulkinta_Bc_edits_clip.shp",
        area="kb3_area.shp",
        image="KB3.png",
        data_type=DataType.TRAINING,
    ),
    Data(
        name="kb7",
        traces="KB7_tulkinta.shp",
        area="kb7_area.shp",
        image="KB7.png",
        data_type=DataType.TRAINING,
    ),
    Data(
        name="kb9",
        traces="KB9_tulkinta_clip1.shp",
        area="kb9_area.shp",
        image="KB9.png",
        data_type=DataType.VALIDATION,
    ),
    Data(
        name="kl2_1",
        traces="KL2_1_tulkinta_clip.shp",
        area="kl2_1_area.shp",
        image="KL2_1.png",
        data_type=DataType.TRAINING,
    ),
    Data(
        name="kl5",
        traces="KL5_tulkinta.shp",
        area="kl5_area.shp",
        image="KL5.png",
        data_type=DataType.TRAINING,
    ),
    Data(
        name="og1",
        traces="OG1_tulkinta.shp",
        area="og1_area.shp",
        image="OG1.png",
        data_type=DataType.PREDICTION,
    ),
]


def _load_traces(reproduction_dir_path: Path):
    traces_zip_path = reproduction_dir_path / DOWNLOADS_DIR_NAME / TRACES_ZIP_NAME
    traces_zip_path.parent.mkdir(parents=True, exist_ok=True)
    traces_dir_path = reproduction_dir_path / DOWNLOADS_DIR_NAME / TRACES_DIR_NAME
    subprocess.check_call(
        ["wget", "--continue", TRACES_URL, f"--output-document={traces_zip_path}"]
    )
    if traces_dir_path.exists():
        shutil.rmtree(traces_dir_path)
    traces_dir_path.mkdir()
    shutil.unpack_archive(filename=traces_zip_path, extract_dir=traces_dir_path)

    src_traces_path = traces_dir_path / TRACES_DIR_TRACES
    src_area_path = traces_dir_path / TRACES_DIR_AREAS

    out_training_traces_path, out_training_area_path = [
        reproduction_dir_path / path for path in (TRAINING_TRACES, TRAINING_AREAS)
    ]
    out_validation_traces_path, out_validation_area_path = [
        reproduction_dir_path / path for path in (VALIDATION_TRACES, VALIDATION_AREAS)
    ]
    out_prediction_traces_path, out_prediction_area_path = [
        reproduction_dir_path / path for path in (PREDICTION_TRACES, PREDICTION_AREAS)
    ]

    for path in (
        out_training_traces_path,
        out_training_area_path,
        out_validation_traces_path,
        out_validation_area_path,
        out_prediction_traces_path,
        out_prediction_area_path,
    ):
        path.mkdir(parents=True, exist_ok=True)

    for data in TRACE_AREA_IMAGE_DATA_PAIRS:
        if data.data_type is DataType.TRAINING:
            out_traces_path = out_training_traces_path
            out_area_path = out_training_area_path
        elif data.data_type is DataType.VALIDATION:
            out_traces_path = out_validation_traces_path
            out_area_path = out_validation_area_path
        elif data.data_type is DataType.PREDICTION:
            out_traces_path = out_prediction_traces_path
            out_area_path = out_prediction_area_path
        else:
            raise ValueError(f"Expected one of DataType enum values.")

        for attr, src_path, out_path in zip(
            ("traces", "area"),
            (
                src_traces_path,
                src_area_path,
            ),
            (
                out_traces_path,
                out_area_path,
            ),
        ):
            glob_pattern = f"{Path(getattr(data, attr)).stem}.*"
            # print(dict(glob_pattern=glob_pattern, src_path=src_path, out_path=out_path))
            # Copy traces
            for shp_file_part in src_path.glob(glob_pattern):
                out_file_path = out_path / shp_file_part.name.lower()
                out_file_path.unlink(missing_ok=True)
                shutil.copy(src=shp_file_part, dst=out_file_path)


def _load_orthomosaics(reproduction_dir_path: Path):
    images_zip_path = reproduction_dir_path / DOWNLOADS_DIR_NAME / IMAGES_ZIP_NAME
    images_zip_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["wget", "--continue", IMAGES_URL, f"--output-document={images_zip_path}"]
    )

    images_dir_path = reproduction_dir_path / DOWNLOADS_DIR_NAME / IMAGES_DIR_NAME
    images_dir_path.mkdir(exist_ok=True, parents=True)
    shutil.unpack_archive(filename=images_zip_path, extract_dir=images_dir_path)

    out_training_images_path = reproduction_dir_path / TRAINING_IMAGES
    out_validation_images_path = reproduction_dir_path / VALIDATION_IMAGES
    out_prediction_images_path = reproduction_dir_path / PREDICTION_IMAGES

    for data in TRACE_AREA_IMAGE_DATA_PAIRS:
        if data.data_type is DataType.TRAINING:
            out_image_path = out_training_images_path
        elif data.data_type is DataType.VALIDATION:
            out_image_path = out_validation_images_path
        elif data.data_type is DataType.PREDICTION:
            out_image_path = out_prediction_images_path
        else:
            raise ValueError(f"Expected DataType enum type.")

        src_image_path = images_dir_path / "pngs" / data.image
        out_image_path.mkdir(exist_ok=True, parents=True)

        # Match from the start of the image name until the first underscore
        src_image_path_base_match = re.match(
            pattern=r"^(\w\w\d+).*", string=src_image_path.stem
        )
        assert src_image_path_base_match is not None
        src_image_path_base = src_image_path_base_match.groups()[0]
        dst_image_stem = src_image_path_base.lower()
        dst_image_name = f"{dst_image_stem}{src_image_path.suffix}"
        dst = out_image_path / dst_image_name

        # print("Preparing to move image:")
        # print(
        #     dict(
        #         src_image_path=src_image_path,
        #         src_image_path_base=src_image_path_base,
        #         dst=dst,
        #     )
        # )

        print(f"Copying {src_image_path} to {dst} and overwriting if necessary.")
        dst.unlink(missing_ok=True)
        shutil.copy(src=src_image_path, dst=dst)
        # src_image_path.unlink(missing_ok=True)


def _load_model(reproduction_dir_path: Path):
    model_output_path = reproduction_dir_path / "unet_weights_2022_a.hdf5"
    subprocess.check_call(
        ["wget", "--continue", MODEL_URL, f"--output-document={model_output_path}"]
    )


def main():
    """
    Reproduce the data for training, validation and prediction phases.

    All options default to `True` i.e. all data is downloaded if no options are
    given.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce the data for training, validation and prediction phases."
    )
    parser.add_argument(
        "--load-traces", action="store_true", help="Load traces", default=True
    )
    parser.add_argument("--no-load-traces", action="store_false", dest="load_traces")
    parser.add_argument(
        "--load-orthomosaics",
        action="store_true",
        help="Load orthomosaics",
        default=True,
    )
    parser.add_argument(
        "--no-load-orthomosaics", action="store_false", dest="load_orthomosaics"
    )
    parser.add_argument(
        "--load-model", action="store_true", help="Load model", default=True
    )
    parser.add_argument("--no-load-model", action="store_false", dest="load_model")
    parser.add_argument(
        "reproduction_dir_path", type=str, help="Path to the reproduction directory"
    )

    args = parser.parse_args()
    if args.load_traces:
        _load_traces(reproduction_dir_path=Path(args.reproduction_dir_path))
    if args.load_orthomosaics:
        _load_orthomosaics(reproduction_dir_path=Path(args.reproduction_dir_path))
    if args.load_model:
        _load_model(reproduction_dir_path=Path(args.reproduction_dir_path))


if __name__ == "__main__":
    main()
