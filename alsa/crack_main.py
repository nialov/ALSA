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

import alsa.geo_proc as gp
import alsa.image_proc as ip
import alsa.signal_proc as sp
from alsa.crack_cls import CrackNetWork
from alsa.data import saveResult, testGenerator
from alsa.model import unet


def crack_main(
    work_dir: Path,
    img_path: Path,
    area_shp_file_path: Path,
    unet_weights_path: Path,
    new_shp_path: Path,
):
    sub_imgs = ip.open_image(img_path)
    orig_dims = sub_imgs.shape

    geo_data = gpd.read_file(area_shp_file_path)
    min_x, min_y, max_x, max_y = geo_data.total_bounds

    w, h = (256, 256)
    sub_imgs = ip.img_segmentation(sub_imgs, width=w, height=h)
    n_mats_per_row = int(orig_dims[1] / w) + 1
    n_mats_per_col = int(orig_dims[0] / h) + 1
    n_mats = n_mats_per_row * n_mats_per_col

    # dirs = [work_dir / dir_name for dir_name in ("sub_imgs", "predictions")]
    sub_imgs_dir = work_dir / "sub_imgs"
    predictions_dir = work_dir / "predictions"
    for dir_path in (sub_imgs_dir, predictions_dir):
        dir_path.mkdir(exist_ok=True, parents=True)

    redundant_id_list = list()
    for i, im in enumerate(sub_imgs):
        if np.quantile(im, 0.95) == 0 or np.quantile(im, 0.05) == 255:
            redundant_id_list.append(i)
        # im_path = dirs[0] + "/sub_img_" + str(i) + ".png"
        img_path = sub_imgs_dir / f"sub_img_{i}.png"
        ip.save_image(img_path, im)

    model = unet(unet_weights_path)
    img_generator = testGenerator(sub_imgs_dir, num_image=n_mats)
    results = model.predict_generator(img_generator, n_mats, verbose=1)
    saveResult(predictions_dir, results)

    nworks = list()
    print("creating nworks")
    for i in range(n_mats):
        print(str(i) + "/" + str(n_mats - 1))
        if i not in redundant_id_list:
            im_path = predictions_dir / f"{i}_predict.png"
            # im_path = dirs[1] + "/" + str(i) + "_predict.png"
            # coords, _ = sp.ridge_fit(im_path, os.getcwd(), img_shape=(w, h))
            coords, _ = sp.ridge_fit(im_path, saved_img_dir=work_dir, img_shape=(w, h))
            nwork = CrackNetWork(coords)
            nwork.connect()
            nwork.remove_small()
            if len(nwork.line_segments) == 0:
                nwork = None
        else:
            nwork = None

        nworks.append(nwork)
    print("combining nworks")
    nworks = CrackNetWork.combine_nworks(nworks, (w, h), n_mats_per_row)
    gdf = nworks.to_geodataframe(orig_dims, (min_x, min_y, max_x, max_y))
    gp.to_shp(gdf, file_path=new_shp_path)

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
