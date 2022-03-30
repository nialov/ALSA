"""
Main entrypoint for prediction using trained model.

Proposed improvements:

-   Create a parametrization for the connecting line which is solely
    used to compare and decide which connector should
    be in the CrackNetWork.connect
-   Create a method for eliminating the case where a line segment
    crosses another one more than once.
-   Specify in CrackNetWork.connect when to use exact angle
    difference calculations
-   Parameter optimization
-   Improve parametrization functions to better emphasize on finding
    the correct angle and less on the distance


"""
from pathlib import Path

import geopandas as gpd
import numpy as np

import alsa.image_proc as ip
import alsa.signal_proc as sp
from alsa.crack_cls import CrackNetWork
from alsa.data import saveResult, testGenerator
from alsa.model import unet


def crack_main(
    work_dir: Path,
    img_path: Path,
    area_file_path: Path,
    unet_weights_path: Path,
    predicted_output_path: Path,
    width: int = 256,
    height: int = 256,
    override_ridge_configs: dict = dict(),
    verbose: bool = True,
    driver: str = "ESRI Shapefile",
):
    # Open image to predict on
    sub_imgs = ip.open_image(img_path)

    # Get dimensions
    orig_dims = sub_imgs.shape

    # Read target area and resolve rectangular bounds
    geo_data = gpd.read_file(area_file_path)
    min_x, min_y, max_x, max_y = geo_data.total_bounds

    # Segment the image to sub-images
    sub_imgs = ip.img_segmentation(sub_imgs, width=width, height=height)
    n_mats_per_row = int(orig_dims[1] / width) + 1
    n_mats_per_col = int(orig_dims[0] / height) + 1
    n_mats = n_mats_per_row * n_mats_per_col

    # Create sub-image and prediction directories in working directory
    sub_imgs_dir = work_dir / "sub_imgs"
    predictions_dir = work_dir / "predictions"
    for dir_path in (sub_imgs_dir, predictions_dir):
        dir_path.mkdir(exist_ok=True, parents=True)

    redundant_id_list = list()

    # Enumerate over the images
    for i, im in enumerate(sub_imgs):
        if np.quantile(im, 0.95) == 0 or np.quantile(im, 0.05) == 255:
            # Mark too homogeneous images for skipping in prediction
            redundant_id_list.append(i)

        # Save sub-images
        img_path = sub_imgs_dir / f"sub_img_{i}.png"
        ip.save_image(img_path, im)

    # Initiate model from given weights
    model = unet(unet_weights_path)

    # Create image generator
    img_generator = testGenerator(sub_imgs_dir, num_image=n_mats)

    # Create predictions for each sub-image
    results = model.predict_generator(img_generator, n_mats, verbose=1)

    # Save prediction images to predictions dir
    saveResult(predictions_dir, results)

    # Start post-processing (ridge detection) from the predicted data
    nworks = list()

    if verbose:
        print("Starting ridge detection from predicted trace data.")

    # Report progress at set intervals (25%, 50%, ...)
    quarter = n_mats // 4
    report_indexes = set(range(0, 5 * quarter, quarter))

    for i in range(n_mats):

        if i in report_indexes and verbose:
            progress = i / n_mats
            print(f"Progress at {int(progress) * 100} %")

        # Default value
        nwork = None

        if i not in redundant_id_list:
            # Not too homogeneous based on earlier examination
            im_path = predictions_dir / f"{i}_predict.png"

            # Fit coordinates with ridge detection
            coords, _ = sp.ridge_fit(
                im_path,
                saved_img_dir=work_dir,
                img_shape=(width, height),
                override_ridge_configs=override_ridge_configs,
            )

            # Create CrackNetWork and process detected into vector traces
            nwork = CrackNetWork(coords)
            nwork.connect()
            nwork.remove_small()
            if len(nwork.line_segments) == 0:
                nwork = None

        nworks.append(nwork)

    if verbose:
        print("Starting combination of CrackNetWorks.")

    # Combine the networks/traces from each sub-image
    nworks = CrackNetWork.combine_nworks(nworks, (width, height), n_mats_per_row)

    # Convert to geopandas.GeoDataFrame
    gdf = nworks.to_geodataframe(orig_dims, (min_x, min_y, max_x, max_y))

    # Save to wanted spatial format based on driver (e.g. ESRI Shapefile)
    gdf.to_file(predicted_output_path, driver=driver)
    if verbose:
        print(f"Saved predicted traces to {predicted_output_path}")

    return nworks, orig_dims, geo_data, gdf


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
