"""
Geometry and GIS related utilities.
"""
import logging
import math
import os
from typing import Callable, List, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString
from sklearn.preprocessing import minmax_scale

from alsa.crack_maths import line_point_generator, linear_converter


# Returns GeoDataFrame from shapely file in geo_path
def geo_data(geo_path):
    return gpd.GeoDataFrame.from_file(geo_path)


def to_shp(gdf, file_path=None):
    if file_path is None:
        file_path = os.getcwd() + "/ACD_GDF.shp"

    gdf.to_file(file_path)


def geo_dataframe_to_list(data_frame, polygon=False):
    """
    Return the GeoDataFrame geometry coordinates as list.

    >>> from shapely.geometry import LineString
    >>> geo_dataframe_to_list(gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])]))
    [[(0.0, 0.0), (1.0, 1.0)]]
    """
    to_return = []
    for line in data_frame.geometry:
        lines = []
        if polygon:
            for values in line.exterior.coords:
                lines.append(values)
        else:
            for values in line.coords:
                lines.append(values)
        to_return.append(lines)
    return to_return


def normalize_slack(unit_vector: np.ndarray, slack: int) -> int:
    """
    Normalize slack based on unit vector of line.
    """
    sqrt_two = 1 / np.sqrt(2)
    max_val = abs(abs(sqrt_two) + abs(sqrt_two))
    min_val = 1
    a = unit_vector[0]
    b = unit_vector[1]
    diff = abs(abs(a) + abs(b))
    inverse_weight = minmax_scale(
        [min_val, diff, max_val], feature_range=(1.0, np.sqrt(2))
    )[1]
    normed_slack = slack / inverse_weight
    normed_slack_int = math.ceil(normed_slack)
    return normed_slack_int


def determine_real_to_pixel_ratio(
    image_shape: Tuple[int, int],
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
):
    """
    Determine ratio to transform real units to pixels.
    """
    image_x = image_shape[1]
    image_y = image_shape[0]
    diff_x = max_x - min_x
    diff_y = max_y - min_y

    resolution_x = image_x / diff_x
    resolution_y = image_y / diff_y

    # Should be similar as cells are rectangles
    if not np.isclose(resolution_x, resolution_y):
        logging.error(
            f"Resolution in x and y axes differ: x: {resolution_x} y: {resolution_y}"
        )

    resolution = np.mean([resolution_x, resolution_y])

    return resolution


def geo_dataframe_to_binmat(
    data_frame: gpd.GeoDataFrame,
    dims,
    relative_geo_data: gpd.GeoDataFrame,
    slack: int = 0,
):
    """
    Return a binary matrix where 1s correspond to the coordinates in the dataframe.

    Slack determines how accurately the pixels in return array are turned into 1s.
    I.e. it is equivalent to a buffer around each trace.
    The higher abs(slack) the more "rounded" the areas.

    >>> from shapely.geometry import LineString, box
    >>> geo_dataframe_to_binmat(gpd.GeoDataFrame(geometry=[LineString([(0, 0), (5, 5)])]), dims=(10, 10), relative_geo_data=gpd.GeoSeries(box(0, 0, 10, 10)))
    array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])
    """

    def append_coord_list(
        coord_list: List[Tuple[int, int]],
        line_string: LineString,
        convert_x: Callable,
        convert_y: Callable,
    ):
        x_list = line_string.xy[0]
        y_list = line_string.xy[1]

        for (x, y) in zip(x_list, y_list):
            x = int(convert_x(x))
            y = int(convert_y(y))
            coord_list.append((x, y))

        return coord_list

    to_return = np.zeros(dims)
    min_x, min_y, max_x, max_y = relative_geo_data.total_bounds

    assert min_x < max_x
    assert min_y < max_y

    convert_x = linear_converter((min_x, max_x), (0, dims[1]), ignore_errors=True)
    convert_y = linear_converter((min_y, max_y), (dims[0], 0), ignore_errors=True)

    for line in data_frame.geometry:
        if line is None:
            continue
        coord_list = list()
        # if line.type == "MultiLineString":
        if isinstance(line, MultiLineString):
            for line_s in line.geoms:
                append_coord_list(
                    coord_list, line_s, convert_x=convert_x, convert_y=convert_y
                )
        elif isinstance(line, LineString):
            append_coord_list(
                coord_list, line, convert_x=convert_x, convert_y=convert_y
            )
        else:
            raise TypeError(
                f"Expected geometries to be (Multi)LineStrings. Got: {type(line)}"
            )

        # (Multi)LineStrings consist of segments between coordinate points
        # Iterate through each segment start and end point:
        for i, crd2 in enumerate(coord_list):
            # crd2 is the end point
            if i == 0:
                continue
            # crd1 is the start point
            crd1 = coord_list[i - 1]

            # Normalize the slack based on the orientation of the line
            # Vertical or horizontal line slack will not be effected
            # but all others are lowered based on the orientation.
            vector = np.array([crd2[1] - crd1[1], crd2[0] - crd1[0]])
            unit_vector = vector / np.linalg.norm(vector)
            normed_slack = normalize_slack(unit_vector=unit_vector, slack=slack)

            # Generate points along a segment
            for x, y in line_point_generator(crd1, crd2):
                try:
                    for x_s in range(-normed_slack, normed_slack + 1):
                        for y_s in range(-normed_slack, normed_slack + 1):
                            result_y = y + y_s
                            result_x = x + x_s
                            if result_y < 0 or result_x < 0:
                                continue
                            to_return[result_y, result_x] = 1
                except IndexError:
                    continue

    return to_return
