"""
Signal processing.
"""
import collections.abc
import logging
from typing import Tuple

import numpy as np
from ridge_detection.helper import displayContours, save_to_disk
from ridge_detection.lineDetector import LineDetector
from ridge_detection.params import Params
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel

import alsa.image_proc as ip

DEFAULT_RIDGE_CONFIGS = {
    # "path_to_file": img_path,
    "mandatory_parameters": {
        # "Sigma": 3.39,
        # "Lower_Threshold": 0.34,
        # "Upper_Threshold": 1.02,
        "Maximum_Line_Length": 0,
        "Minimum_Line_Length": 0,
        "Darkline": "LIGHT",
        "Overlap_resolution": "NONE",
    },
    "optional_parameters": {
        "Line_width": 3,
        "High_contrast": 200,
        "Low_contrast": 60,
    },
    "further_options": {
        "Correct_position": True,
        "Estimate_width": True,
        "doExtendLine": True,
        "Show_junction_points": True,
        "Show_IDs": False,
        "Display_results": False,
        "Preview": False,
        "Make_Binary": True,
        "save_on_disk": True,
    },
}


def update(d: dict, u: dict):
    """
    Update dictionary d with values from dictionary u.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode="wrap")
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


# Returns the kernels for the gabor filter with theta values theta_0 to theta_n with n steps (np.linspace)
def gabor_kernels(freq, n=1, theta_0=0, theta_n=np.pi, bw=1):
    kernels = []

    if theta_0 == 0 and theta_n == np.pi:
        theta_n = theta_n - theta_n / n

    for i in np.linspace(theta_0, theta_n, n):
        kernel = np.real(gabor_kernel(freq, theta=i, bandwidth=bw))
        kernels.append(kernel)
    relic = int(round(1 / freq))
    return kernels, relic


def power(image, kernel, relic):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    mat = np.sqrt(
        ndi.convolve(image, np.real(kernel), mode="wrap") ** 2
        + ndi.convolve(image, np.imag(kernel), mode="wrap") ** 2
    )
    if relic < min(image.shape[0] / 2, image.shape[1] / 2):
        mat = mat[relic:(-relic), relic:(-relic)]
    mat_min = np.min(mat)
    mat_max = np.max(mat)

    if mat_max == 0:
        raise Exception("Image is an array of zeros")
    else:
        mat = (mat - mat_min) / mat_max * 255
    return mat


# returns the gabor filtered image used by kernels
def gabor_image(image, kernels, relic):
    results = []
    for k in kernels:
        results.append(power(image, k, relic))
    return results


# Returns the linear combination of the images in image_list with
# weights in weights is None, all weights are equal to 1.
def lin_comb(image_list, weights=None):
    array_to_return = np.zeros(image_list[0].shape)
    if weights is None:
        for im in image_list:
            array_to_return += im
    elif len(weights) == len(image_list):
        for im, weight in zip(image_list, weights):
            array_to_return += weight * im
    else:
        raise Exception("len of weights differs from len of image_list")
    array_to_return = array_to_return / len(image_list)
    return array_to_return


# Attempts to denoise the array by taking the average of given
# square amount of pixels
def avg_denoise(array, pixels=3):
    row_start = 0
    col_start = 0
    to_return = np.copy(array)
    while col_start < (array.shape[1] + pixels):
        row_start = 0
        while row_start < (array.shape[0] + pixels):
            to_return[
                row_start : (row_start + pixels), col_start : (col_start + pixels)
            ] = np.mean(
                array[
                    row_start : (row_start + pixels), col_start : (col_start + pixels)
                ]
            )
            row_start += pixels
        col_start += pixels
    return to_return


def resolve_ridge_config(img_path: str, override_ridge_configs: dict) -> dict:
    """
    Resolve final ridge detection configuration.
    """
    ridge_configs = DEFAULT_RIDGE_CONFIGS
    ridge_configs["path_to_file"] = img_path

    ridge_configs = update(ridge_configs, override_ridge_configs)
    return ridge_configs


def ridge(
    img_path, saved_img_dir, override_ridge_configs: dict, save_on_file: bool = False
):

    ridge_configs = resolve_ridge_config(
        img_path=img_path, override_ridge_configs=override_ridge_configs
    )

    params = Params(ridge_configs)

    img = ip.open_image(img_path, asarray=False)

    detect = LineDetector(params=ridge_configs)
    try:
        result = detect.detectLines(img)
    except (SystemExit, NameError) as exc:
        logging.error(
            f"Caught {exc} exception from ridge detection. img_path = {img_path}",
        )
        return [], None
    resultJunction = detect.junctions

    out_img, img_only_lines = displayContours(params, result, resultJunction)
    if save_on_file:
        save_to_disk(out_img, img_only_lines, str(saved_img_dir))

    # result has the coordinates, img_only_lines is an Image
    return result, img_only_lines


# attempts to form straight lines or line segments out of line
# objects calculated by ridge function
# returns a list of list of tuples of len 2 which contain the
# coordinates for the line segment points in a matrix
def calculate_ridge_points(line_o, max_length=80):
    # returns list of list of tuples
    def calculate_accurate_points(
        angle_list, line_o, orig_size, start_point=0, x_range=(0, 255), y_range=(0, 255)
    ):
        def extend_line(coord_list, x_range, y_range):
            x_d = coord_list[0][1][0] - coord_list[0][0][0]
            y_d = coord_list[0][1][1] - coord_list[0][0][1]
            x_orig = coord_list[0][1][0]
            y_orig = coord_list[0][1][1]

            x_1 = x_orig
            x_2 = x_orig
            y_2 = y_orig
            y_1 = y_orig

            if x_d != 0:
                k = y_d / x_d

                while x_1 >= np.min(line_o.col) and y_2 >= np.min(line_o.row):
                    x_1 -= 1
                    y_2 -= k

                    min_d = np.inf
                    for x, y in zip(line_o.col, line_o.row):
                        current_d = np.sqrt((x - x_1) ** 2 + (y - y_2) ** 2)
                        if current_d < min_d:
                            min_d = current_d
                    if min_d > 10:
                        break

                while x_2 <= np.max(line_o.col) and y_1 <= np.max(line_o.row):
                    x_2 += 1
                    y_1 += k

                    min_d = np.inf
                    for x, y in zip(line_o.col, line_o.row):
                        current_d = np.sqrt((x - x_2) ** 2 + (y - y_1) ** 2)
                        if current_d < min_d:
                            min_d = current_d
                    if min_d > 10:
                        break

                y_1 = int(round(min(y_range[1] - 1, max(y_range[0], y_1))))
                y_2 = int(round(min(y_range[1] - 1, max(y_range[0], y_2))))
                x_1 = int(round(min(x_range[1] - 1, max(x_range[0], x_1))))
                x_2 = int(round(min(x_range[1] - 1, max(x_range[0], x_2))))

                return (x_1, y_2), (x_2, y_1)
            else:
                while y_2 > np.min(y_range):
                    y_2 -= 1
                while y_1 < np.max(y_range):
                    y_1 += 1

                return (x_1, y_2), (x_2, y_1)

        n_points = line_o.num
        median_angle = np.median(angle_list)
        first_p = 0
        last_p = len(angle_list) - 1
        min_first = min_last = np.inf
        for i in range(int(len(angle_list) / 3)):
            if abs(angle_list[i] - median_angle) < min_first:
                first_p = i
                min_first = abs(angle_list[i] - median_angle)
            if abs(angle_list[-i - 1] - median_angle) < min_last:
                last_p = len(angle_list) - i - 1
                min_last = abs(angle_list[-i - 1] - median_angle)
        first_p = int((first_p + start_point) / orig_size * (n_points - 1))
        last_p = int((last_p + start_point) / orig_size * (n_points - 1))
        to_return = [
            [
                (line_o.col[first_p], line_o.row[first_p]),
                (line_o.col[last_p], line_o.row[last_p]),
            ]
        ]

        a, b = extend_line(to_return, x_range, y_range)
        to_return = [a, b]
        return [to_return]

    # attempts to form a line segment and return the index of the
    # coordinate in the line which should be chosen
    def ridge_segmentation(angle_list, line_o, n_points, orig_size, start=0):

        mid_p = int((len(angle_list)) / 2)
        angle1 = angle_list[:mid_p]
        angle2 = angle_list[mid_p:]
        to_return = list()

        angle1_std = np.std(angle1)
        angle2_std = np.std(angle2)
        line1_list = list()
        line2_list = list()
        if angle1_std <= 0.2 or mid_p < 21:
            x_min = int(start / orig_size * n_points)
            x_max = int((start + mid_p) / orig_size * n_points)
            x_range_0 = np.min(line_o.col[x_min:x_max])
            x_range_1 = np.max(line_o.col[x_min:x_max])
            y_range_0 = np.min(line_o.row[x_min:x_max])
            y_range_1 = np.max(line_o.row[x_min:x_max])
            line1_list = calculate_accurate_points(
                angle1,
                line_o,
                orig_size,
                start_point=start,
                x_range=(x_range_0, x_range_1),
                y_range=(y_range_0, y_range_1),
            )
        else:
            to_return.extend(
                ridge_segmentation(angle1, line_o, n_points, orig_size, start=start)
            )

        if len(line1_list) > 0:
            to_return.extend(line1_list)

        if angle2_std <= 0.2 or mid_p < 21:
            x_max = int((start + len(angle_list)) / orig_size * n_points)
            x_min = int((start + mid_p) / orig_size * n_points)
            x_range_0 = np.min(line_o.col[x_min:x_max])
            x_range_1 = np.max(line_o.col[x_min:x_max])
            y_range_0 = np.min(line_o.row[x_min:x_max])
            y_range_1 = np.max(line_o.row[x_min:x_max])
            line2_list = calculate_accurate_points(
                angle2,
                line_o,
                orig_size,
                start_point=int((start + mid_p) / orig_size * n_points),
                x_range=(x_range_0, x_range_1),
                y_range=(y_range_0, y_range_1),
            )
        else:
            to_return.extend(
                ridge_segmentation(
                    angle2, line_o, n_points, orig_size, start=mid_p + start
                )
            )

        if len(line2_list) > 0:
            to_return.extend(line2_list)
        return to_return

    n_points = line_o.num
    orig_size = len(line_o.angle)
    # if line is small enough, the end points will suffice
    if n_points >= 20:
        angles = line_o.angle
        angles_std = np.std(angles)
        # check if the angle of the pixels are consistent
        if angles_std < 0.2 and len(line_o.col) <= max_length:
            return calculate_accurate_points(
                angles,
                line_o,
                orig_size,
                x_range=(np.min(line_o.col), np.max(line_o.col)),
                y_range=(np.min(line_o.row), np.max(line_o.row)),
            )
        else:
            # check if the angle in the middle is consistent
            quartile = int(len(angles) / 4)
            quartile_angles = angles[quartile : (len(angles) - quartile)]
            angles_std = np.std(quartile_angles)
            if angles_std < 0.2 and len(line_o.col) <= max_length:
                cut_point = int(n_points / 4)
                return calculate_accurate_points(
                    quartile_angles,
                    line_o,
                    orig_size,
                    cut_point,
                    x_range=(np.min(line_o.col), np.max(line_o.col)),
                    y_range=(np.min(line_o.row), np.max(line_o.row)),
                )
            else:
                to_return = list()
                for line in ridge_segmentation(angles, line_o, n_points, orig_size):
                    line_list = list()
                    for i in line:
                        line_list.append(i)
                    to_return.append(line_list)
            return to_return
    else:
        return [[(line_o.col[0], line_o.row[0]), (line_o.col[-1], line_o.row[-1])]]


def ridge_fit(
    img_path,
    saved_img_dir,
    override_ridge_configs: dict,
    img_shape: Tuple[int, int] = (256, 256),
    slack: int = 1,
    save_on_file: bool = False,
):
    coords, _ = ridge(
        img_path=img_path,
        saved_img_dir=saved_img_dir,
        override_ridge_configs=override_ridge_configs,
        save_on_file=save_on_file,
    )
    line_coords_list = list()
    img_list = list()
    for _, line in enumerate(coords):
        line_coords = calculate_ridge_points(line)
        im = ip.img_binmat_line_segment(line_coords, img_shape, slack=slack)
        line_coords_list.append(line_coords)
        img_list.append(im)

    return line_coords_list, img_list


# attempts to extrapolate lines in ridge_fit_coords by checking for
# lines with same position and angle
# DEPRAVED!! DO NOT USE
def connect_lines(ridge_fit_coords, eps=3, shape=(256, 256)):
    line_list = list()

    # extrapolate line until meets the edge of the image
    def extrapolate(coord, k, shape):
        x_orig = coord[0]
        y_orig = coord[1]

        x_1 = x_orig
        x_2 = x_orig
        y_2 = y_orig
        y_1 = y_orig

        while x_1 > 0 and y_2 > 0:
            x_1 -= 1
            y_2 -= k

        while x_2 < shape[1] - 1 and y_1 < shape[0] - 1:
            x_2 += 1
            y_1 += k

        y_1 = int(round(min(255, max(0, y_1))))
        y_2 = int(round(min(255, max(0, y_2))))

        return (x_1, y_2), (x_2, y_1)

    # return the coordinates of a line when extrapolated to the
    # edge of the image
    def edge_coords(line_coords, shape):
        x_d = line_coords[1][0] - line_coords[0][0]
        y_d = line_coords[1][1] - line_coords[0][1]
        if x_d > 0:
            k = y_d / x_d
            a, b = extrapolate(line_coords[0], k, shape)
            return [a, b]

        else:
            return [(line_coords[0][0], 0), (line_coords[0][0], shape[0] - 1)]

    def combine_lines(edge_points, line_list, eps):
        def compare_coords(crds1, crds2, eps):

            satisfied = True

            for c1, c2 in zip(crds1, crds2):
                if abs(c1[0] - c2[0]) > eps or abs(c1[1] - c2[1]) > eps:
                    satisfied = False

            return satisfied

        def distance(crds1, crds2):
            def eukl(x1, x2, y1, y2):
                return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            distances = list()

            for crd1 in crds1:
                for crd2 in crds2:
                    distances.append(eukl(crd1[0], crd2[0], crd1[1], crd2[1]))
            return np.min(distances)

        to_return = list()
        sorted_list = list()
        for i, points in enumerate(edge_points):
            id_list = list()
            id_list.append(i)
            sorted_list.append(i)
            for j, points2 in enumerate(edge_points):
                if j not in sorted_list:
                    if (
                        compare_coords(points, points2, eps)
                        and distance(line_list[i], line_list[j]) < np.max(shape) / 8
                    ):
                        id_list.append(j)
                        sorted_list.append(j)
            to_return.append(id_list)
        return to_return

    def fill_line(coord_list):
        x_min = np.inf
        x_max = 0
        y_min = y_max = 0

        for crds in coord_list:
            for crd in crds:
                if crd[0] < x_min:
                    x_min = crd[0]
                    y_min = crd[1]
                if crd[0] > x_max:
                    x_max = crd[0]
                    y_max = crd[1]

        return [(x_min, y_min), (x_max, y_max)]

    for line_segment in ridge_fit_coords:
        for line in line_segment:
            line_list.append(line)

    extrapol_list = list()
    for j, line in enumerate(line_list):
        extrapol_list.append(edge_coords(line, shape))

    grouped_ids = combine_lines(extrapol_list, line_list, eps)

    grouped_line_list = list()

    for grp in grouped_ids:
        group = list()
        for id in grp:
            group.append(line_list[id])
        grouped_line_list.append(group)

    for i, grp in enumerate(grouped_line_list):
        new_coords = fill_line(grp)
        grouped_line_list[i] = new_coords

    return grouped_line_list
