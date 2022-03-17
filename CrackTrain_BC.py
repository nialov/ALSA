import glob
import os
import shutil
from pathlib import Path

import geopandas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import geo_proc as gp
import image_proc as ip
import signal_proc as sp
from data import trainGenerator
from model import ModelCheckpoint, unet


# continues training the model with weights in old_weight_path. If None, starts from scratch
# and saves it to working directory.
# New weights will be saved to a file in new_weight_path. If None, it will instead overwrite the old one.
def train_main(work_dir=None, old_weight_path=None, new_weight_path=None):
    orig_img_dir = Path(r"Training/Images/Originals")
    shp_bounds_dir = Path(r"Training/Shapefiles/Areas")
    shp_dir = Path(r"Training/Shapefiles/Labels")
    gen_dir = Path(r"Training/Images/Generated")
    org_val_img_dir = Path(r"Validation\Images\Originals")
    val_img_dir = Path(r"Validation\Images\Generated")
    val_shp_dir = Path(r"Validation\Shapefiles\Labels")
    val_bound_dir = Path(r"Validation\Shapefiles\Areas")
    if new_weight_path is None:
        new_weight_path = Path("unet_weights.hdf5")
    model = unet(old_weight_path)

    images = list(orig_img_dir.glob("*.png"))
    trace_labels = list(shp_dir.glob("*.shp"))
    bounds = list(shp_bounds_dir.glob("*.shp"))
    print(images, trace_labels, bounds)

    training_list = []
    for image in images:
        for label in trace_labels:
            for bound in bounds:
                if image.stem in label.name and image.stem in bound.name:
                    training_list.append((str(image), str(label), str(bound)))
    print(training_list)

    #     def filebrowser(dir, ext=''):
    #         return [f for f in glob.glob(f'{dir}/*{ext}')]

    #     img_names = filebrowser(orig_img_dir, '.png')
    #     shp_names = filebrowser(shp_dir, '.shp')
    #     bound_names = filebrowser(shp_bounds_dir, '.shp')
    #     print(img_names, shp_names, bound_names)
    #     training_list = list()
    #     for i, file in enumerate(img_names):
    #         area_name = file[(len(orig_img_dir) + 1):(len(file) - 4)]
    #         for j, shp in enumerate(shp_names):
    #             shp_name = shp[(len(shp_dir) + 1):(len(shp) - 4)]
    #             if area_name in shp_name:
    #                 for k, ar in enumerate(bound_names):
    #                     ar_name = ar[(len(shp_bounds_dir) + 1):(
    #                                 len(ar) - 4)]
    #                     if area_name in ar_name:
    #                         training_list.append((img_names[i],
    #                                               shp_names[j],
    #                                               bound_names[k]))
    #                         break
    #                 else:
    #                     print(
    #                         f'Could not find a .shp file in {shp_bounds_dir} representing the area in image {file} and .shp file {shp}')
    #                 break
    #         else:
    #             print(
    #                 f'Could not find a .shp file representing image {file}')

    assert len(training_list) > 0, "Could not find any .png-.shp file pairs."

    for ind, (img, geo_file, geo_bounds) in enumerate(training_list):
        img = ip.open_image(img)
        geo_data = geopandas.GeoDataFrame.from_file(geo_file)
        geo_area = geopandas.GeoDataFrame.from_file(geo_bounds)
        label_bin_img = gp.geo_dataframe_to_binmat(
            geo_data, img.shape, relative_geo_data=geo_area, slack=0
        )

        # divide original img (and binary images) into sub images of shape (h, w) and save them
        w, h = (256, 256)
        sub_imgs = ip.img_segmentation(img, width=w, height=h)
        sub_bin_imgs = ip.img_segmentation(label_bin_img, width=w, height=h)
        suf = 1
        for (im, b) in zip(sub_imgs, sub_bin_imgs):
            if np.quantile(im, 0.95) == 0 or np.quantile(im, 0.05) == 255:
                continue
            src_output_path = gen_dir / (
                "Source/src" + str(ind) + "_" + str(suf) + ".png"
            )
            src_output_path.parent.mkdir(parents=True, exist_ok=True)
            ip.save_image(src_output_path, im)
            label_output_path = gen_dir / (
                "Labels/lbl" + str(ind) + "_" + str(suf) + ".png"
            )
            label_output_path.parent.mkdir(parents=True, exist_ok=True)
            ip.save_image(label_output_path, b)
            suf += 1

    # BC-Reading validation images, etc.
    # v_img_names = filebrowser(org_val_img_dir, '.png')
    # v_shp_names = filebrowser(val_shp_dir, '.shp')
    # v_bound_names = filebrowser(val_bound_dir, '.shp')
    val_list = training_list.copy()
    # print(val_list)
    # for i, file in enumerate(v_img_names):
    #     v_area_name = file[(len(org_val_img_dir) + 1):(len(file) - 4)]
    #     for j, shp in enumerate(v_shp_names):
    #         v_shp_name = shp[(len(val_shp_dir) + 1):(len(shp) - 4)]
    #         if v_area_name in v_shp_name:
    #             for k, ar in enumerate(v_bound_names):
    #                 v_ar_name = ar[(len(val_bound_dir) + 1):(
    #                             len(ar) - 4)]
    #                 if v_area_name in v_ar_name:
    #                     val_list.append((v_img_names[i],
    #                                           v_shp_names[j],
    #                                           v_bound_names[k]))
    #                     break
    #             else:
    #                 print(
    #                     f'Could not find a .shp file in {val_bound_dir} representing the area in image {file} and .shp file {shp}')
    #             break
    #     else:
    #         print(
    #             f'Could not find a .shp file representing image {file}')

    assert len(val_list) > 0, "Could not find any .png-.shp file pairs."

    for ind, (img, geo_file, geo_bounds) in enumerate(val_list):
        img = ip.open_image(img)
        geo_data = geopandas.GeoDataFrame.from_file(geo_file)
        geo_area = geopandas.GeoDataFrame.from_file(geo_bounds)
        label_bin_img = gp.geo_dataframe_to_binmat(
            geo_data, img.shape, relative_geo_data=geo_area, slack=0
        )

        # divide original img (and binary images) into sub images of shape (h, w) and save them
        w, h = (256, 256)
        sub_imgs = ip.img_segmentation(img, width=w, height=h)
        sub_bin_imgs = ip.img_segmentation(label_bin_img, width=w, height=h)
        suf = 1
        for (im, b) in zip(sub_imgs, sub_bin_imgs):
            if np.quantile(im, 0.95) == 0 or np.quantile(im, 0.05) == 255:
                continue
            src_output_path = val_img_dir / (
                "Source_v/src_v" + str(ind) + "_" + str(suf) + ".png"
            )
            src_output_path.parent.mkdir(parents=True, exist_ok=True)
            ip.save_image(src_output_path, im)
            val_output_path = val_img_dir / (
                "Labels_v/lbl_v" + str(ind) + "_" + str(suf) + ".png"
            )
            val_output_path.parent.mkdir(parents=True, exist_ok=True)
            ip.save_image(val_output_path, b)
            suf += 1

    # set up the training generator's image altering parameters
    data_gen_args = dict(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    N = 1

    # sv_dir = work_dir + '/Training/Images/' #bc
    train_generator = trainGenerator(
        64, gen_dir, "Source", "Labels", data_gen_args, save_to_dir=None
    )

    val_generator = trainGenerator(
        64, val_img_dir, "Source_v", "Labels_v", data_gen_args, save_to_dir=None
    )

    model_checkpoint = ModelCheckpoint(
        new_weight_path, monitor="val_loss", mode="min", verbose=1, save_best_only=True
    )  # BC monitoring validation loss instead of training loss

    H = model.fit(
        train_generator,
        validation_data=val_generator,
        validation_steps=1,
        steps_per_epoch=1,
        epochs=N,
        callbacks=[model_checkpoint],
    )

    # convert the history.history dict to a pandas DataFrame:
    hist_df_ = pd.DataFrame(H.history)
    # save to csv:
    hist_csv_file = "history.csv"
    with open(hist_csv_file, mode="w") as f:
        hist_df_.to_csv(f)
    # model.evaluate(val_generator)
    # N = 20
    plt.style.use("ggplot")
    plt.figure()

    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("Epoch Number")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(np.arange(0, N), H.history["loss"], label="train_loss", color="green")
    ax1.plot(np.arange(0, N), H.history["val_loss"], label="val_loss", color="red")
    ax1.legend(loc="upper left")
    ax1.tick_params(axis="y", labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color2 = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color2)  # we already handled the x-label with ax1
    ax2.plot(np.arange(0, N), H.history["accuracy"], label="train_acc", color="black")
    ax2.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc", color="blue")
    ax2.tick_params(axis="y", labelcolor=color2)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    pass


# Tries to find necessary files for the training. If fails, it will create them to the working directory
def train_setup(work_dir=None):
    training_dirs = [
        "/Training",
        "/Training/Images",
        "/Training/Shapefiles",
        "/Training/Shapefiles/Areas",
        "/Training/Images/Originals",
        "/Training/Shapefiles/Labels",
        "/Training/Images/Generated",
        "/Training/Images/Generated/Source",
        "/Training/Images/Generated/Predictions",
        "/Training/Images/Generated/Labels",
        "/Validation",
        "/Validation/Images",
        "/Validation/Shapefiles",
        "/Validation/Shapefiles/Areas",
        "/Validation/Images/Originals",
        "/Validation/Shapefiles/Labels",
        "/Validation/Images/Generated",
        "/Validation/Images/Generated/Source_v",
        "/Validation/Images/Generated/Predictions",
        "/Validation/Images/Generated/Labels_v",
    ]
    for dir in training_dirs:
        if dir == "/Training/Images/Generated":
            try:
                shutil.rmtree(work_dir + dir)
            except FileNotFoundError:
                pass
        try:
            os.makedirs(work_dir + dir)
        except OSError:
            pass

    pass


if __name__ == "__main__":
    work_dir = os.getcwd()
    train_setup(work_dir=work_dir)
    train_main(work_dir=work_dir)
