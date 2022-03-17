"""

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
import os

import geopandas

import geo_proc as gp
from crack_cls import *
from data import *
from model import *


def crack_main(img_path, area_shp_file_path, unet_weights_path, new_shp_path):
    sub_imgs = ip.open_image(img_path)
    orig_dims = sub_imgs.shape

    geo_data = geopandas.GeoDataFrame.from_file(area_shp_file_path)
    bounds = gp.geo_bounds(geo_data, polygon=True)
    min_x, min_y, max_x, max_y = (
        bounds.minx[0],
        bounds.miny[0],
        bounds.maxx[0],
        bounds.maxy[0],
    )

    w, h = (256, 256)
    sub_imgs = ip.img_segmentation(sub_imgs, width=w, height=h)
    n_mats_per_row = int(orig_dims[1] / w) + 1
    n_mats_per_col = int(orig_dims[0] / h) + 1
    n_mats = n_mats_per_row * n_mats_per_col

    dirs = ["sub_imgs", "predictions"]
    for i, dir in enumerate(dirs):
        dirs[i] = os.getcwd() + "/" + dir
        try:
            os.makedirs(dirs[i])
        except OSError:
            pass

    redundant_id_list = list()
    for i, im in enumerate(sub_imgs):
        if np.quantile(im, 0.95) == 0 or np.quantile(im, 0.05) == 255:
            redundant_id_list.append(i)
        im_path = dirs[0] + "/sub_img_" + str(i) + ".png"
        ip.save_image(im_path, im)

    model = unet(unet_weights_path)
    img_generator = testGenerator(dirs[0], num_image=n_mats)
    results = model.predict_generator(img_generator, n_mats, verbose=1)
    saveResult(dirs[1], results)

    nworks = list()
    print("creating nworks")
    for i in range(n_mats):
        print(str(i) + "/" + str(n_mats - 1))
        if i not in redundant_id_list:
            im_path = dirs[1] + "/" + str(i) + "_predict.png"
            coords, _ = sp.ridge_fit(im_path, os.getcwd(), img_shape=(w, h))
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

    return nworks, orig_dims, geo_data


if __name__ == "__main__":

    img_path = r"C:\Kallioperan_reco\ALSA-main\KL5_ortho_clip_geom.png"
    shp_path = r"C:\Kallioperan_reco\ALSA-main\KL5_ortho_clip_geom_area.shp"
    weight_path = r"C:\Kallioperan_reco\ALSA-main\unet_weights.hdf5"
    new_shp_path = r"C:\Kallioperan_reco\ALSA-main\Test_shp"

    ## BC removed the prompt to direct paths of the input
    """
    def get_input(prompt_str):
        to_return = None
        while True:
            to_return = input(prompt_str)
            try:
                to_return = glob.glob(to_return)[0]
                if not os.getcwd() in to_return:
                    to_return = os.getcwd() + '/' + to_return
                break
            except IndexError:
                try:
                    to_return = \
                        glob.glob(os.getcwd() + '/' + to_return)[0]
                    break
                except IndexError:
                    print(f'Failed to find file {to_return}.')
                    continue
        return to_return
    """
    """
    print(f'Current working directory is {os.getcwd()}')
    img_path = get_input(
        'Please enter either the full or relative path of the image to be analyzed: ')
   
    shp_path = get_input(
       'Please enter either the full or relative path of the shapefile of the area to be analyzed: ')
    
    weight_path = get_input(
        'Please enter either the full or relative path of the unet weight folder: ')
    
    new_shp_path = input(
        'Please enter the name of the output file: ')
    """
    """
    try:
        if new_shp_path[(len(new_shp_path) - 4):] != '.shp':
            new_shp_path = new_shp_path + '.shp'
    except IndexError:
        new_shp_path = new_shp_path + '.shp'

    new_shp_path = os.getcwd() + '/' + new_shp_path
    """
    nworks, orig_dims, geo_data = crack_main(
        img_path, shp_path, weight_path, new_shp_path
    )

    pass
