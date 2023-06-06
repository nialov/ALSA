"""
Image related utilities.
"""
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.segmentation import random_walker
from skimage.transform import AffineTransform, rotate, warp


# Opens an image in img_path as an array (or as an Image file if asarray = False)
def open_image(img_path, show_img=False, greyscale=True, asarray=True):
    with Image.open(img_path) as kuva:
        kuva.load()
        if greyscale:
            kuva = kuva.convert(mode="L")

        if show_img:
            kuva.show()

        if asarray:
            kuva = np.array(kuva)
        return kuva


def save_image(img_path, img):
    if np.max(img) > 1:
        if np.max(img) == 0:
            raise Exception("attempted to save an empty image")
        array = img / np.max(img)
    else:
        array = np.copy(img)
    if len(img.shape) == 2:
        new_img = np.zeros((img.shape[0], img.shape[1], 3))
        new_img[:, :, 0] = array
        new_img[:, :, 1] = array
        new_img[:, :, 2] = array
        plt.imsave(img_path, new_img)
    else:
        plt.imsave(img_path, array)


# Function tries to remove an even coloured background from the image
def crop_image(array, bckgrnd_pxl=0):
    def crop_height(array, bckgrnd_pxl):
        row = 0
        rows = array.shape[0]
        cols = array.shape[1]

        top_found = False
        bot_found = False
        while not (top_found and row < (rows - 1)):
            for point in range(cols):
                if abs(array[row, point] - bckgrnd_pxl) > 120:
                    array = array[row : (rows - 1), :]
                    top_found = True
                    break

            row += 1
        rows = array.shape[0]
        row = rows - 1

        while not (bot_found and row > 1):
            for point in range(cols):
                if abs(array[row, point] - bckgrnd_pxl) > 120:
                    array = array[0:row, :]
                    bot_found = True
                    break
            row -= 1
        return array

    array = crop_height(array, bckgrnd_pxl=bckgrnd_pxl)
    array = np.transpose(array)
    array = crop_height(array, bckgrnd_pxl=bckgrnd_pxl)
    array = np.transpose(array)
    return array


# Plots a given greyscale array or a list of arrays
def plot_array(array, max_cols=5):
    if isinstance(array, list):
        nrow = 1
        # Calculate the no. of rows
        for i in range(1, int(round(np.sqrt(len(array)))) + 1):
            if len(array) % i == 0:
                nrow = i
        ncol = int(len(array) / nrow)
        if nrow == 1 and ncol > max_cols:
            ncol = max_cols
            nrow = int(len(array) / ncol) + 1

        fig, axes = plt.subplots(nrows=nrow, ncols=ncol)
        plt.gray()
        if ncol == nrow == 1:
            axes.imshow(array[0] * 255 / np.max(array[0]))
        elif nrow == 1:
            for mat, ax in zip(array, axes):
                ax.imshow(mat * 255 / np.max(mat))
                ax.axis("off")

        else:
            array_list = []
            alist = []
            for row in range(nrow):
                for col in range(ncol):
                    try:
                        alist.append(array[row * ncol + col])
                    except IndexError:
                        break
                array_list.append(alist)
                alist = []

            axes[0][0].axis("off")
            for mat_row, ax_row in zip(array_list, axes):
                for mat, ax in zip(mat_row, ax_row):
                    ax.imshow(mat * 255 / np.max(mat))
                    ax.axis("off")
        plt.tight_layout()
        plt.show()
        pass

    else:
        fig, axes = plt.subplots(nrows=1, ncols=1)
        plt.gray()
        axes.axis("off")
        axes.imshow(array)
        plt.tight_layout()
        plt.show()

        pass


# Returns data for comparing images: intersect/originalarea, 2*intersect/(original + new area), false positive rate, false negative rate
def compare_binaries(x, original_bin):
    img_pos_points = 0
    x_true_pos = 0
    x_false_pos = 0
    x_true_neg = 0
    x_false_neg = 0

    for line_x, line_data in zip(x, original_bin):
        for point, data in zip(line_x, line_data):
            if point == data:
                if point > 0:
                    x_true_pos += 1
                    img_pos_points += 1
                else:
                    x_true_neg += 1
            else:
                if point > 0:
                    x_false_pos += 1
                else:
                    x_false_neg += 1
                    img_pos_points += 1

    return (
        x_true_pos / img_pos_points,
        2 * x_true_pos / (x_true_pos + x_false_pos + img_pos_points),
        x_false_pos / (x_false_pos + x_true_neg),
        x_false_neg / (x_false_neg + x_true_pos),
    )


# Applies random walker method to data and returns it as a binary image
def img_random_walker(data, lower_q=0.9, upper_q=0.98, val_beta=10, val_mode="bf"):
    labels = []
    if not isinstance(data, list):
        raise Exception("Argument data must be a list type")
    for x in data:
        ones = np.ones(x.shape)
        markers = np.zeros(x.shape)
        markers[x < np.quantile(x, lower_q)] = 1
        markers[x > np.quantile(x, upper_q)] = 2

        labels.append(random_walker(x, markers, beta=val_beta, mode=val_mode) - ones)

    return labels


# Takes a list of binary matrices and returns the union of them
def img_binary_union(data):
    to_return = np.zeros(data[0].shape)
    for x in data:
        to_return += x

    to_return[to_return > 0.0] = 1

    return to_return


# Divide the array into a list of arrays with given widths and heights.
# The arrays are returned as a list, with the top left corner of the array being the first item, then width amount to
# the right from that next, and so on.
# if the next potential sub-array doesn't have enough rows or columns, they are instead embedded into the last one
# method has to be either 'fill' or 'append'
# if 'append', the method attempts to add the remaining rows and/or columns to the current matrix, if the next one
# would be too small in either of those dimensions
# if 'fill', the method creates a 256x256 matrix and if dimensions exceed the original array, fills the rest with 0s
def img_segmentation(img, width=256, height=256, method="fill"):
    def check_next_boundary(boundary, current, step):
        if (current + 2 * step) > boundary:
            return boundary
        else:
            return current + step

    background = 0 if (np.mean(img[:3, :]) + np.mean(img[:, :3])) / 2 < 123 else 255

    if method == "fill":
        rows, cols = img.shape
        row_splits = rows / height
        col_splits = cols / width

        if not row_splits.is_integer():
            row_splits = int(row_splits) + 1
        else:
            row_splits = int(row_splits)
        if not col_splits.is_integer():
            col_splits = int(col_splits) + 1
        else:
            col_splits = int(col_splits)

        array = np.zeros((height * row_splits, width * col_splits)) + background
        array[:rows, :cols] = img

    elif method == "append":
        array = np.zeros(img.shape) + background
        array[:, :] = img[:, :]
    else:
        raise ValueError('Argument "method" must be either "fill" or "append"')

    rows, cols = array.shape

    row_splits = rows / height
    col_splits = cols / width

    row_splits = int(row_splits)
    col_splits = int(col_splits)

    for row in range(row_splits):
        next_row = check_next_boundary(rows, height * row, height)
        for col in range(col_splits):
            next_col = check_next_boundary(cols, width * col, width)

            yield array[(height * row) : next_row, (width * col) : next_col]


# averages the values in the given bin_array_list
def img_bin_brush(bin_array_list, pixels=1):
    to_return = []
    for bin_array in bin_array_list:
        array = np.copy(bin_array)
        row_n = 0
        for row in array:
            col_n = 0
            for val in row:
                if val == np.max(bin_array):
                    left_l = max(0, col_n - pixels)
                    right_l = min(bin_array.shape[1], col_n + pixels + 1)
                    upper_l = max(0, row_n - pixels)
                    lower_l = min(bin_array.shape[0], row_n + pixels + 1)
                    array[upper_l:lower_l, left_l:right_l] = np.mean(
                        array[upper_l:lower_l, left_l:right_l]
                    )
                col_n += 1
            row_n += 1
        to_return.append(array)
    return to_return

    pass


def img_ridge_coords_to_binmat(ridge_results, shape=(256, 256), mat_per_line=False):
    if mat_per_line:
        to_return = list()
    else:
        to_return = np.zeros(shape=shape)
    for line in ridge_results:
        if mat_per_line:
            line_mat = np.zeros(shape=shape)
            for row_val, col_val in zip(line.row, line.col):
                line_mat[row_val, col_val] = 1
            to_return.append(line_mat)
        else:
            for row_val, col_val in zip(line.row, line.col):
                to_return[row_val, col_val] = 1

    return to_return


def img_assemble(array_list, n_mats_per_row, n_mats_per_col):
    dims = None
    for i, mat in enumerate(array_list):
        if dims is None:
            dims = mat.shape
            to_return = np.zeros((n_mats_per_col * dims[0], n_mats_per_row * dims[1]))
        mat_col_pos = (i % n_mats_per_row) * dims[1]
        mat_row_pos = int(i / n_mats_per_row) * dims[0]
        to_return[
            mat_row_pos : (mat_row_pos + dims[0]), mat_col_pos : (mat_col_pos + dims[1])
        ] = mat

    return to_return


def img_rotate(img, angle=90):
    return rotate(img, angle=angle)


def img_warp(img, translation=(-50, -50)):
    transform = AffineTransform(translation=translation)
    warp_image = warp(img, transform, mode="wrap")

    return warp_image


# appends augmented images to the lists x_train, y_train and returns the lists
def img_augment(x_train, y_train, alt_img_ratio):
    x_return = list(range(len(x_train)))
    y_return = list(range(len(y_train)))
    assert alt_img_ratio >= 0, "alt_img_ratio should be either 0 or higher"
    amount = int(alt_img_ratio)
    p = alt_img_ratio - amount

    for i, (x, y) in enumerate(zip(x_train, y_train)):
        extra = 0
        x_return[i] = x
        y_return[i] = y
        if random.random() > p:
            extra = 1
        for i in range(amount + extra):
            p_a = random.random()
            assigned = False
            if p_a >= 0.25:
                rot_angle = random.random() * 180 - 90
                x_new = img_rotate(x, angle=rot_angle)
                y_new = img_rotate(y, angle=rot_angle)
                assigned = True
            if (1 - p_a) >= 0.25:
                max_warp_1 = x.shape[0] / 2
                max_warp_2 = x.shape[1] / 2
                max_warp = (max_warp_1, max_warp_2)
                warp_amount = (
                    random.random() * max_warp[0],
                    random.random() * max_warp[1],
                )
                if assigned:
                    x_new = img_warp(x_new, warp_amount)
                    y_new = img_warp(y_new, warp_amount)
                else:
                    x_new = img_warp(x, warp_amount)
                    y_new = img_warp(y, warp_amount)
            for row in x_new:
                for col in row:
                    if col > 0:
                        col == 1
            for row in y_new:
                for col in row:
                    if col > 0:
                        col == 1

            x_return.append(x_new)
            y_return.append(y_new)
    return x_return, y_return


# takes in 3 greyscale arrays: prediction, original and label. Returns RGB version of orig and label with the pixels in
# pred colored as hl_col


def img_pred_highlight(pred, orig=None, label=None, hl_col=(1, 0, 0)):
    if orig is not None:
        assert pred.shape == orig.shape, "pred and orig must be of same shape"
        orig_b = True
    else:
        orig_b = False

    if label is not None:
        assert pred.shape == label.shape, "pred and label must be of same shape"
        label_b = True
    else:
        label_b = False

    if not (orig_b or label_b):
        raise Exception("orig and label must not both be None")

    def highlight(p, o, col):
        cl_img = np.zeros((p.shape[0], p.shape[1], 3))

        (cl_red, cl_grn, cl_blue) = col

        cl_img[:, :, 0] = p * cl_red
        cl_img[:, :, 1] = p * cl_grn
        cl_img[:, :, 2] = p * cl_blue

        orig_m = np.zeros(o.shape)
        for row in range(orig_m.shape[0]):
            for col in range(orig_m.shape[1]):
                if p[row, col] < 255:
                    orig_m[row, col] = o[row, col]

        cl_img[:, :, 0] = cl_img[:, :, 0] + orig_m
        cl_img[:, :, 1] = cl_img[:, :, 1] + orig_m
        cl_img[:, :, 2] = cl_img[:, :, 2] + orig_m

        return cl_img

    if orig_b:
        cl_orig = highlight(pred, orig, hl_col)

    if label_b:
        cl_label = highlight(pred, label, hl_col)

    if orig_b and label_b:
        return cl_orig, cl_label
    elif orig_b:
        return cl_orig
    else:
        return cl_label


# takes in a list of tuples which should have values that within the dimensions of the matrix shape
# returns a binary matrix where the coordinates in line_coords have been joined forming line segments with width slack
def img_binmat_line_segment(line_coords, shape=(256, 256), slack=0):
    to_return = np.zeros(shape=shape)

    for line in line_coords:
        previous_coord = -1
        for coord in line:
            to_return[coord[1], coord[0]] = 1

            if previous_coord == -1:
                previous_coord = coord
                continue
            else:
                x_d = abs(coord[1] - previous_coord[1])
                y_d = abs(coord[0] - previous_coord[0])
                if x_d == 0 and y_d == 0:
                    break
                n_steps = 2 * max(x_d, y_d)

                x_steps = np.linspace(coord[1], previous_coord[1], n_steps)
                y_steps = np.linspace(coord[0], previous_coord[0], n_steps)

                for x_step, y_step in zip(y_steps, x_steps):
                    try:
                        to_return[int(y_step), int(x_step)] = 1
                        if slack > 0:
                            to_return[
                                int(max(0, min(shape[0] - 1, y_step - slack))) : int(
                                    max(0, min(shape[0] - 1, y_step + slack))
                                ),
                                int(max(0, min(shape[1] - 1, x_step - slack))) : int(
                                    max(0, min(shape[1] - 1, x_step + slack))
                                ),
                            ] = 1

                    except IndexError:
                        continue

                previous_coord = coord

    return to_return
