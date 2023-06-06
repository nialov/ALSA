"""
Main entrypoint for prediction using trained model.
"""
import json
import logging
import math
from functools import partial
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np

import alsa.image_proc as ip
import alsa.signal_proc as sp
from alsa import utils
from alsa.crack_cls import CrackNetWork
from alsa.data import save_result, testGenerator
from alsa.model import unet

RIDGE_CONFIG_PATH = Path("ridge_config.json")
SUB_IMGS_PATH = "sub_imgs"
PREDICTIONS_PATH = "predictions"


def resolve_ridge_config_overrides(
    override_ridge_config_path: Optional[Path],
    work_dir: Path,
) -> dict:
    """
    Resolve ridge config override dictionary.

    If a ``Path`` is inputted, tries to load json from that path. If None is
    given, tries to find config in default location in ``work_dir`` and load
    the json from there.
    """
    if override_ridge_config_path is None:
        override_ridge_config_path = work_dir / RIDGE_CONFIG_PATH

    if override_ridge_config_path.exists():
        override_ridge_config = json.loads(override_ridge_config_path.read_text())
    else:
        logging.info(f"Found no ridge config overrides at {override_ridge_config_path}")
        override_ridge_config = dict()
    assert isinstance(override_ridge_config, dict)
    return override_ridge_config


def crack_main(
    work_dir: Path,
    img_path: Path,
    area_file_path: Path,
    unet_weights_path: Path,
    predicted_output_path: Path,
    override_ridge_config: Union[dict, Optional[Path]] = None,
    width: int = 256,
    height: int = 256,
    verbose: bool = True,
    driver: str = "ESRI Shapefile",
):
    print_if_verbose = partial(utils.conditional_print, is_verbose=verbose)

    # Resolve overrides to ridge-detection
    if isinstance(override_ridge_config, Path) or override_ridge_config is None:
        override_ridge_config = resolve_ridge_config_overrides(
            override_ridge_config_path=override_ridge_config, work_dir=work_dir
        )
    assert isinstance(override_ridge_config, dict)

    # Open image to predict on
    predict_image = ip.open_image(img_path)

    print_if_verbose(f"Opening image for prediction {img_path}")

    # Get dimensions
    orig_dims = predict_image.shape
    print_if_verbose(f"Image dimensions: {orig_dims}")

    # Read target area and resolve rectangular bounds
    geo_data = gpd.read_file(area_file_path)
    min_x, min_y, max_x, max_y = geo_data.total_bounds
    print_if_verbose(
        f"Read target area from {area_file_path}. Bounds: {min_x, min_y, max_x, max_y}."
    )

    # Segment the image to sub-images
    sub_imgs = ip.img_segmentation(predict_image, width=width, height=height)
    n_mats_per_row = int(orig_dims[1] / width) + 1
    n_mats_per_col = int(orig_dims[0] / height) + 1
    n_mats = n_mats_per_row * n_mats_per_col
    segment_info = dict(
        n_mats_per_row=n_mats_per_row, n_mats_per_col=n_mats_per_col, n_mats=n_mats
    )
    print_if_verbose(f"Segmenting image. {segment_info}")

    # Create sub-image and prediction directories in working directory
    sub_imgs_dir = work_dir / SUB_IMGS_PATH
    predictions_dir = work_dir / PREDICTIONS_PATH
    for dir_path in (sub_imgs_dir, predictions_dir):
        dir_path.mkdir(exist_ok=True, parents=True)

    print_if_verbose(
        f"{dict(sub_imgs_dir=sub_imgs_dir, predictions_dir=predictions_dir)}"
    )
    redundant_id_list = list()

    # Enumerate over the images
    sub_imgs_list = list(sub_imgs)
    for i, im in enumerate(sub_imgs_list):
        if np.quantile(im, 0.95) == 0 or np.quantile(im, 0.05) == 255:
            # Mark too homogeneous images for skipping in prediction
            redundant_id_list.append(i)
            logging.warning(f"Image {im} at index {i} is too homogeneous.")

        # Save sub-images
        img_path = sub_imgs_dir / f"sub_img_{i}.png"
        ip.save_image(img_path, im)

    # Report or raise an error if homogeneous images are too large a proportion
    # of all images
    utils.report_redundant_proportion(
        redundant_proportion=len(redundant_id_list) / len(sub_imgs_list)
    )

    # Initiate model from given weights
    model = unet(unet_weights_path)
    print_if_verbose(f"Model loaded from {unet_weights_path}")

    # Create image generator
    img_generator = testGenerator(sub_imgs_dir, num_image=n_mats)

    # Create predictions for each sub-image
    results = model.predict_generator(img_generator, n_mats, verbose=1)

    # Save prediction images to predictions dir
    save_result(
        predictions_dir,
        results,
        n_mats_per_col=n_mats_per_col,
        n_mats_per_row=n_mats_per_row,
    )
    print_if_verbose(
        f"Saving prediction images. {dict(predictions_dir=predictions_dir)}"
    )

    # Start post-processing (ridge detection) from the predicted data
    nworks = list()

    print_if_verbose("Starting ridge detection from predicted trace data.")

    # Report progress at set intervals (25%, 50%, ...)
    quarter = n_mats // 4
    report_indexes = set(range(0, 5 * quarter, quarter))

    for i in range(n_mats):
        if i in report_indexes:
            progress = i / n_mats
            print_if_verbose(f"Progress at {math.ceil(progress * 100)} %")

        # Default value
        nwork = None

        if i not in redundant_id_list:
            # Not too homogeneous based on earlier examination
            # (not in redundant_id_list)
            im_path = predictions_dir / f"{i}_predict.png"

            # Fit coordinates with ridge detection
            coords, _ = sp.ridge_fit(
                im_path,
                saved_img_dir=work_dir,
                img_shape=(width, height),
                override_ridge_config=override_ridge_config,
            )

            # Create CrackNetWork and process detected into vector traces
            nwork = CrackNetWork(coords)
            nwork.connect()
            nwork.remove_small()
            if len(nwork.line_segments) == 0:
                nwork = None

        nworks.append(nwork)

    print_if_verbose("Starting combination of CrackNetWorks.")

    # Combine the networks/traces from each sub-image
    combined_nwork = CrackNetWork.combine_nworks(
        nworks, (width, height), n_mats_per_row
    )
    assert isinstance(combined_nwork, CrackNetWork)

    # Convert to geopandas.GeoDataFrame
    gdf = combined_nwork.to_geodataframe(orig_dims, (min_x, min_y, max_x, max_y))
    assert isinstance(gdf, gpd.GeoDataFrame)

    # Save to wanted spatial format based on driver (e.g. ESRI Shapefile)
    if not gdf.empty:
        gdf.to_file(predicted_output_path, driver=driver)
        print_if_verbose(f"Saved predicted traces to {predicted_output_path}")
    else:
        logging.error(
            "Empty GeoDataFrame from prediction.\n"
            f"image: {img_path}\n weights: {unet_weights_path}"
        )

    return combined_nwork, orig_dims, geo_data, gdf


# if __name__ == "__main__":

#     img_path = Path("Training/Images/Originals/kl5_subsample_circle.png")
#     shp_path = Path("Training/Shapefiles/Areas/kl5_subsample_circle.shp")
#     weight_path = Path("unet_weights.hdf5")
#     new_shp_path = Path("outputs")

#     ## BC removed the prompt to direct paths of the input
#     # nworks, orig_dims, geo_data = crack_main(
#     #     img_path, shp_path, weight_path, new_shp_path
#     # )
#     crack_main(
#         img_path, shp_path, weight_path, new_shp_path
#     )
