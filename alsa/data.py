"""
Data utilities.
"""
import glob
import os
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import skimage.io as io
import skimage.transform as trans
from tensorflow.keras.preprocessing.image import ImageDataGenerator

Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array(
    [
        Sky,
        Building,
        Pole,
        Road,
        Pavement,
        Tree,
        SignSymbol,
        Fence,
        Car,
        Pedestrian,
        Bicyclist,
        Unlabelled,
    ]
)

MOSAIC_PREDICT_PATH = Path("mosaic_predict.png")


def lin_normalize(array):
    """
    Normalize image array.

    >>> lin_normalize(np.array([[1, 3], [3, 5], [5, 7], [7, 9], [9, 11]]))
    array([[  0,  51],
           [ 51, 102],
           [102, 153],
           [153, 204],
           [204, 255]])
    """
    min_val = np.min(array)
    max_val = np.max(array)
    if min_val == max_val:
        return np.zeros(array.shape)
    d_min = 0
    d_max = 255

    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            val = array[row, col]
            array[row, col] = (d_max - d_min) * (val - min_val) / (
                max_val - min_val
            ) + d_min

    return array


def adjustData(img, mask, flag_multi_class, num_class):
    if flag_multi_class:
        img = img / 255
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            # for one pixel in the image, find the class in mask and convert it into one-hot vector
            # index = np.where(mask == i)
            # index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            # new_mask[index_mask] = 1
            new_mask[mask == i, i] = 1
        new_mask = (
            np.reshape(
                new_mask,
                (
                    new_mask.shape[0],
                    new_mask.shape[1] * new_mask.shape[2],
                    new_mask.shape[3],
                ),
            )
            if flag_multi_class
            else np.reshape(
                new_mask, (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2])
            )
        )
        mask = new_mask
    elif np.max(img) > 1:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)


def trainGenerator(
    batch_size,
    train_path,
    image_folder,
    mask_folder,
    aug_dict,
    image_color_mode="grayscale",
    mask_color_mode="grayscale",
    image_save_prefix="image",
    mask_save_prefix="mask",
    flag_multi_class=False,
    num_class=2,
    save_to_dir=None,
    target_size=(256, 256),
    seed=1,
):
    """
    Get train generator.

    Can generate image and mask at the same time.
    Use the same seed for image_datagen and mask_datagen to
    ensure the transformation for image and mask is the same
    if you want to visualize the results of generator,
    set save_to_dir = "your path"
    """
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
    )
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
    )
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)


def testGenerator(
    test_path,
    num_image=30,
    target_size=(256, 256),
    flag_multi_class=False,
    as_gray=True,
):
    for i in range(num_image):
        # img = io.imread(os.path.join(test_path,"sub_img_%d.png"%i),as_gray = as_gray)
        img = io.imread(test_path / f"sub_img_{i}.png", as_gray=as_gray)
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def geneTrainNpy(
    image_path,
    mask_path,
    flag_multi_class=False,
    num_class=2,
    image_prefix="image",
    mask_prefix="mask",
    image_as_gray=True,
    mask_as_gray=True,
):
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(
            item.replace(image_path, mask_path).replace(image_prefix, mask_prefix),
            as_gray=mask_as_gray,
        )
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def save_result(
    save_path: Path,
    array: np.ndarray,
    n_mats_per_row: int,
    n_mats_per_col: int,
    flag_multi_class: bool = False,
    num_class: int = 2,
    normalize: bool = True,
    norm_func: Callable = lin_normalize,
):
    assert len(array) == n_mats_per_row * n_mats_per_col
    assert isinstance(array, np.ndarray)

    # Gather predicted sub-images to a dictionary
    # Results are an array with a dimension of 1
    # however the results can be parsed back together
    # into a single mosaic
    row_idx = 1
    mosaic_dict: Dict[int, List[np.ndarray]] = dict()

    # Iterate over results
    for i, item in enumerate(array):

        if i == n_mats_per_row * row_idx:
            row_idx += 1

        assert isinstance(item, np.ndarray)
        img = (
            labelVisualize(num_class, COLOR_DICT, item)
            if flag_multi_class
            else item[:, :, 0]
        )
        assert isinstance(img, np.ndarray)
        if normalize and (norm_func is not None):
            img = norm_func(img)
        img = img.astype("uint8")
        # io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        img_path = save_path / f"{i}_predict.png"

        # Save predicted sub-image
        io.imsave(img_path, img)

        # Gather predicted sub-image for concatenation into mosaic later
        if row_idx in mosaic_dict:
            mosaic_dict[row_idx].append(img)
        else:
            mosaic_dict[row_idx] = list()

    # Create mosaic of sub-images
    rows = []
    for row in mosaic_dict.values():
        rows.append(np.concatenate(row, axis=1))
    mosaic_array = np.concatenate(rows)

    # Save mosaic
    mosaic_path = save_path / MOSAIC_PREDICT_PATH
    io.imsave(mosaic_path, mosaic_array)
