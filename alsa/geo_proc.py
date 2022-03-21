"""
Geometry and GIS related utilities.
"""
import os

import geopandas
import numpy as np

from alsa.crack_maths import line_point_generator, linear_converter


# Returns GeoDataFrame from shapely file in geo_path
def geo_data(geo_path):
    return geopandas.GeoDataFrame.from_file(geo_path)


def to_shp(gdf, file_path=None):
    if file_path is None:
        file_path = os.getcwd() + "/ACD_GDF.shp"

    gdf.to_file(file_path)


def geo_dataframe_to_list(data_frame, polygon=False):
    """
    Return the GeoDataFrame geometry coordinates as list.

    >>> import geopandas as gpd
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


def geo_bounds(data_frame, polygon=False):
    """
    Return the minimum and maximum x and y coordinates in the given GeoDataFrame.

    >>> import geopandas as gpd
    >>> from shapely.geometry import LineString
    >>> geo_bounds(gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])]))
    (0.0, 0.0, 1.0, 1.0)
    """
    bound_list = []

    if polygon:
        return data_frame.bounds
    else:
        for line in data_frame.geometry:
            bound_list.append(line.bounds)

    x_min = bound_list[0][0]
    y_min = bound_list[0][1]
    x_max = bound_list[0][2]
    y_max = bound_list[0][3]
    for boundary in bound_list:
        if boundary[0] < x_min:
            x_min = boundary[0]
        if boundary[1] < y_min:
            y_min = boundary[1]
        if boundary[2] > x_max:
            x_max = boundary[2]
        if boundary[3] > y_max:
            y_max = boundary[3]

    return x_min, y_min, x_max, y_max


def geo_normalize(
    data_frame,
    x_min=0,
    y_min=0,
    x_max=1,
    y_max=1,
    polygon=False,
    relative_geo_data=None,
):
    """
    Normalize the coordinates in the GeoDataFrame into the given range and return it as a list.

    TODO: Need to check what this actually does...
    """
    if polygon:
        bounds = data_frame.geometry[0].bounds
    else:
        bounds = geo_bounds(relative_geo_data, polygon=False)
    x_range = bounds[2] - bounds[0]
    y_range = bounds[3] - bounds[1]

    crd_list = geo_dataframe_to_list(data_frame, polygon=polygon)
    to_return = []

    for line in crd_list:
        lines = []
        for point in line:
            crds = []
            x_pass = y_pass = False
            x_c = (x_max - x_min) * (point[0] - bounds[0]) / x_range + x_min
            if x_min <= x_c <= x_max:
                crds.append(x_c)
                x_pass = True
            y_c = (y_min - y_max) * (point[1] - bounds[1]) / y_range + y_max

            if y_min <= y_c <= y_max:
                crds.append(y_c)
                y_pass = True

            if x_pass and y_pass:
                lines.append(crds)
        to_return.append(lines)
    return to_return


def geo_dataframe_to_binmat(data_frame, dims, relative_geo_data, slack=0):
    """
    Return a binary matrix where 1s correspond to the coordinates in the dataframe.

    Slack determines how accurately the pixels in return array are turned into 1s.
    The higher abs(slack) the more "rounded" the areas.

    >>> from shapely.geometry import LineString, box
    >>> import geopandas as gpd
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
    to_return = np.zeros(dims)
    bounds = relative_geo_data.bounds
    min_x, min_y, max_x, max_y = (
        bounds.minx[0],
        bounds.miny[0],
        bounds.maxx[0],
        bounds.maxy[0],
    )

    convX = linear_converter((min_x, max_x), (0, dims[1]), ignore_errors=True)
    convY = linear_converter((min_y, max_y), (dims[0], 0), ignore_errors=True)

    def append_coord_list(coord_list, line_string):
        x_list = line_string.xy[0]
        y_list = line_string.xy[1]

        for (x, y) in zip(x_list, y_list):
            x = int(convX(x))
            y = int(convY(y))
            coord_list.append((x, y))

        return coord_list

    for line in data_frame.geometry:
        if line is None:
            continue
        coord_list = list()
        if line.type == "MultiLineString":
            for line_s in line:
                append_coord_list(coord_list, line_s)
        else:
            append_coord_list(coord_list, line)

        for i, crd2 in enumerate(coord_list):
            if i == 0:
                continue
            crd1 = coord_list[i - 1]
            for x, y in line_point_generator(crd1, crd2):
                try:
                    for x_s in range(-slack, slack + 1):
                        for y_s in range(-slack, slack + 1):
                            to_return[y + y_s, x + x_s] = 1
                except IndexError:
                    continue

    return to_return
