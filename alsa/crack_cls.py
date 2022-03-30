"""
Implementation of CrackNetwork.
"""
from math import isclose

import geopandas
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

import alsa.image_proc as ip
from alsa.crack_maths import linear_converter


def crack_eukl(crd1, crd2):
    """
    Calculate euclidean distance.

    >>> round(crack_eukl(np.array([0, 0]), np.array([1, 1])), 3)
    1.414
    """
    return np.sqrt((crd1[0] - crd2[0]) ** 2 + (crd1[1] - crd2[1]) ** 2)


def crack_sigmoid(func):
    def sigmoid_wrapper(distance_val, mp=1 / 42, c=-4, *args, **kwargs):
        distance_val = 1 - 1 / (1 + np.exp(-distance_val * mp + c))
        return func(distance_val=distance_val, *args, **kwargs)

    return sigmoid_wrapper


# returns a value between 0 and 1 which attempts to estimate the probability where two lines with
# angle x should be connected
def crack_pw_linear(func):
    def linear_wrapper(
        angle_val,
        end_points=(0, np.pi / 2 - np.pi / 18),
        mid_points=(np.pi / 6, np.pi / 3),
        *args,
        **kwargs,
    ):
        if mid_points[0] < angle_val < mid_points[1]:
            angle_val = 0.5

        elif angle_val <= mid_points[0]:
            angle_val = 1 - 0.5 / (mid_points[0] - end_points[0]) * angle_val
        else:
            angle_val = 1 - (
                min(
                    1,
                    (angle_val - mid_points[1]) * 0.5 / (end_points[1] - mid_points[1])
                    + 0.5,
                )
            )

        return func(angle_val=angle_val, *args, **kwargs)

    return linear_wrapper


# 1/np.sqrt(2)
# returns the parameter for deciding wether lines should be connected with given probabilities
# based on angle and distance
@crack_sigmoid
@crack_pw_linear
def crack_connection_parameter(angle_val, distance_val, weights=(1, 1)):
    return np.sqrt(
        weights[0] * (1 - angle_val) ** 2 + weights[1] * (1 - distance_val) ** 2
    )


class Crack:
    def __init__(self, end_points):
        assert isinstance(end_points, list), "Crack must be instantiated with a list"
        assert len(end_points) == 2, "Crack must be a list of len 2"
        assert isinstance(end_points[0], tuple) and isinstance(
            end_points[1], tuple
        ), "List elements must be tuples in Crack"
        end_points = sorted(end_points, key=lambda tup: (tup[0], tup[1]))
        self.end_points = end_points

    def __getitem__(self, item):
        return self.end_points[item]

    def __str__(self):
        return (
            "("
            + str(self.end_points[0][0])
            + ", "
            + str(self.end_points[0][1])
            + ") ("
            + str(self.end_points[1][0])
            + ", "
            + str(self.end_points[1][1])
            + ")"
        )

    def __add__(self, other):
        assert isinstance(
            other, Crack
        ), "operator + is only defined between two instances of Crack"
        to_return = Crack(
            sorted([self.position, other.end_points[1]], key=lambda x: (x[0], x[1]))
        )
        return to_return

    def __eq__(self, other):
        assert isinstance(
            other, Crack
        ), "operator == can only be done between two instances of Crack"

        to_return = True

        for p1, p2 in zip(self.end_points, other.end_points):
            if p1 != p2:
                to_return = False
                break
        return to_return

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def angle(self):
        x_d = self[1][0] - self[0][0]

        if x_d > 0:
            y_d = self[0][1] - self[1][1]
            k = y_d / x_d
        elif x_d == 0:
            k = np.inf
        else:
            ValueError(
                "the coordinates in {} have been sorted incorrectly".format(self)
            )

        return np.arctan(k)

    @property
    def slope(self):
        return -np.tan(self.angle)

    @property
    def normal_slope(self):
        return np.cos(self.angle), np.sin(self.angle)

    @property
    def position(self):
        return self.end_points[0]

    @property
    def right_angle(self):
        if self.angle > 0:
            return self.angle - np.pi / 2
        else:
            return self.angle + np.pi / 2

    @property
    def length(self):
        return crack_eukl(self.end_points[0], self.end_points[1])

    @staticmethod
    def bounds(*line_lists):
        def compare(current, line):
            min_x = current[0]
            min_y = current[1]
            max_x = current[2]
            max_y = current[3]
            if min(line.end_points[0][0], line.end_points[1][0]) < min_x:
                min_x = line.end_points[0][0]
            if max(line.end_points[0][0], line.end_points[1][0]) > max_x:
                max_x = line.end_points[1][0]
            if min(line.end_points[0][1], line.end_points[1][1]) < min_y:
                min_y = min(line.end_points[0][1], line.end_points[1][1])
            if max(line.end_points[0][1], line.end_points[1][1]) > max_y:
                max_y = max(line.end_points[0][1], line.end_points[1][1])

            return min_x, min_y, max_x, max_y

        min_x = min_y = np.inf
        max_x = max_y = 0

        for l in line_lists:
            for line in l:
                if line is None:
                    continue
                min_x, min_y, max_x, max_y = compare((min_x, min_y, max_x, max_y), line)
        return (min_x, min_y), (max_x, max_y)

    def sort(self):
        self.end_points = sorted(self.end_points, key=lambda x: (x[0], x[1]))

    # check if the crack contains a coordinate
    def contains(self, coord):
        def points(start, stop, normal_slope):
            curr_pos = start
            while curr_pos[0] <= stop[0]:
                yield curr_pos
                new_x = curr_pos[0] + normal_slope[0]
                new_y = curr_pos[1] + normal_slope[1]

                curr_pos = (new_x, new_y)

        previous_d = np.inf
        for pnt in points(self.position, self.end_points[1], self.normal_slope):
            current_d = crack_eukl(coord, pnt)
            if current_d <= 1.5:
                return True
            if previous_d < current_d:
                return False
            else:
                previous_d = current_d

    # extend the crack by a length given
    def extend_by(self, length, dims=None, side=None):
        if dims is None:
            max_x, max_y = np.inf, np.inf
        else:
            (max_y, max_x) = dims
        assert (
            side is None or side == 0 or side == 1
        ), "argument side in extend_by must be either None, 0 or 1"
        y_d = -int(round(np.sin(self.angle) * length))
        x_d = int(round(np.cos(self.angle) * length))
        if side is None or side == 0:
            self.end_points[0] = (
                max(0, self.end_points[0][0] - x_d),
                max(0, self.end_points[0][1] - y_d),
            )

        if side is None or side == 1:
            self.end_points[1] = (
                min(max_x, self.end_points[1][0] + x_d),
                min(max_y, self.end_points[1][1] + y_d),
            )

        return self

    # cuts off the part of the crack that would go beyond the dimensions of a matrix
    def fit_to_mat(self, dims):
        def check_bounds(coord, dims):
            if 0 <= coord[0] <= dims[1] - 1 and 0 <= coord[1] <= dims[0] - 1:
                return True
            else:
                return False

        self.end_points[0] = (int(self.end_points[0][0]), int(self.end_points[0][1]))
        self.end_points[1] = (int(self.end_points[1][0]), int(self.end_points[1][1]))

        out_of_bounds = list()

        for coord in self.end_points:
            if check_bounds(coord, dims):
                continue
            else:
                out_of_bounds.append(coord)

        if len(out_of_bounds) == 0:
            pass
        else:

            x_d = 1
            y_d = self.slope

            for coord in out_of_bounds:
                coord_minus = coord
                coord_plus = coord
                while not (
                    check_bounds(coord_minus, dims) or check_bounds(coord_plus, dims)
                ):
                    coord_minus[0] -= x_d
                    coord_minus[1] -= y_d
                    coord_plus[0] += x_d
                    coord_plus[1] += x_d
                if check_bounds(coord_minus, dims):
                    self.end_points[1] = coord_minus
                else:
                    self.end_points[0] = coord_plus

            self.end_points[0] = (
                int(self.end_points[0][0]),
                int(self.end_points[0][1]),
            )
            self.end_points[1] = (
                int(self.end_points[1][0]),
                int(self.end_points[1][1]),
            )

    # moves the cracks coordinates by an amount
    def move_by(self, amount):
        assert isinstance(
            amount, (tuple, list)
        ), "amount should be either a tuple or a list"
        assert len(amount), "the len of amount should be 2"

        self.end_points[0] = (
            self.end_points[0][0] + amount[0],
            self.end_points[0][1] + amount[1],
        )
        self.end_points[1] = (
            self.end_points[1][0] + amount[0],
            self.end_points[1][1] + amount[1],
        )

    # returns the angle between two cracks
    def angle_difference(self, other, exact=False):

        to_return = abs(self.angle - other.angle)
        if to_return > np.pi / 2:
            to_return = np.pi - to_return
        if not exact:
            return to_return
        else:
            connection = (-1, 0)
            for i, ep1 in enumerate(self.end_points):
                if connection[0] == -1:
                    for j, ep2 in enumerate(self.end_points):
                        if crack_eukl(ep1, ep2) < 1:
                            connection = (i, j)
                            break
                else:
                    break
            else:
                raise AssertionError(f"lines {self} and {other} do not connect")
            if connection[0] == 1:
                to_return = np.pi - to_return
            if connection[1] == 1:
                to_return = np.pi - to_return
            return to_return

    # forms 4 cracks that each connect one end point of self to an end point of other and returns the shortest one
    # if end_points are given, the desired crack must have its end points in end_points, one end in the first element
    # and one end in the second element
    def shortest_connecting_line(self, other, end_points=None):
        if end_points is None:
            current_crack = Crack([self.position, other.position])

            for coord1 in self.end_points:
                for coord2 in other.end_points:
                    new_crack = Crack([coord1, coord2])
                    if new_crack.length < current_crack.length:
                        current_crack = new_crack
            return current_crack
        else:
            end_points1 = end_points[0]
            end_points2 = end_points[1]
            current_crack = None
            for crd1 in self.end_points:
                if crd1 in end_points1:
                    for crd2 in other.end_points:
                        if crd2 in end_points2:
                            if current_crack is None:
                                current_crack = Crack([crd1, crd2])
                            else:
                                new_crack = Crack([crd1, crd2])
                                if new_crack.length < current_crack.length:
                                    current_crack = new_crack
            return current_crack

    # compares the shortest distances between self and all cracks in crack_network and returns the crack that is
    # the shortest among those
    def find_closest_crack(self, crack_network):
        current_crack = None
        for line_s in crack_network:
            previous_crack = current_crack
            for line in line_s:
                if line is self:
                    if line_s.index(line) > 0:
                        current_crack = previous_crack
                    break
                else:
                    new_crack = self.shortest_connecting_line(line)
                    if current_crack is None:
                        current_crack = new_crack
                    elif current_crack.length > new_crack:
                        current_crack = new_crack

        assert current_crack is not None, "failed to find closest crack in a network"
        return current_crack

    # returns a binmat with a shape of dims where the end points and elements between the end points are marked as 1s
    def as_binmat(self, dims=(256, 256)):
        to_return = np.zeros(dims)

        x_d = int(self.end_points[1][0] - self.end_points[0][0])
        y_d = int(self.end_points[1][1] - self.end_points[0][1])

        n_steps = 2 * max(abs(x_d), abs(y_d))

        x_steps = np.linspace(
            int(self.end_points[0][0]), int(self.end_points[1][0]), n_steps
        )
        y_steps = np.linspace(
            int(self.end_points[0][1]), int(self.end_points[1][1]), n_steps
        )

        for x, y in zip(x_steps, y_steps):
            try:
                to_return[int(y), int(x)] = 1
            except IndexError:
                break
        return to_return

    @staticmethod
    def angle_pass(line1, line2, bp=np.pi / 3):
        angle = line1.angle_difference(line2)
        if abs(angle) < bp:
            return True
        else:
            return False

    def mid_point(self):
        return (
            (int((self.end_points[0][0] + self.end_points[1][0]) / 2)),
            (int((self.end_points[0][1] + self.end_points[1][1]) / 2)),
        )

    # returns true if line overlaps with other, false otherwise
    def overlaps_with(self, other):

        k1 = self.slope
        k2 = other.slope

        c1 = self.position[1] - k1 * self.position[0]
        c2 = other.position[1] - k2 * other.position[0]

        to_return = None
        if isclose(k1, k2, abs_tol=10**-3) or (k1 >= 1e3 and k2 >= 1e3):
            to_return = isclose(c1, c2, abs_tol=1e-4)
        elif k1 >= 1e3 and k2 < 1e3:

            # calculate the crossing point
            y = k2 * self.position[0] + c2
            x = self.position[0]

        elif k1 < 1e3 and k2 >= 1e3:

            y = k1 * other.position[0] + c1
            x = other.position[0]

        else:

            x = (c2 - c1) / (k1 - k2)
            y = k1 * x + c1

        if to_return is None:
            # check if crossing point is in both lines
            to_return = self.contains((x, y)) and other.contains((x, y))

        return to_return

    def is_close(self, other, max_d=2):

        d0 = crack_eukl(self.end_points[0], other.end_points[0])
        d1 = crack_eukl(self.end_points[1], other.end_points[1])

        if d1 <= max_d and d0 <= max_d:
            return True
        else:
            return False

    # returns an extrapolated version of self which is extrapolated until it meets other
    # if not possible, returns None
    def extrapolate_to(self, other):

        k1 = self.slope
        k2 = other.slope

        c1 = self.position[1] - k1 * self.position[0]
        c2 = other.position[1] - k2 * other.position[0]

        to_return = None
        if isclose(k1, k2, abs_tol=10**-3) or (k1 >= 1e3 and k2 >= 1e3):
            to_return = None
        elif k1 >= 1e3 and k2 < 1e3:
            to_return = True  # placeholder value for later conditional check
            # calculate the crossing point
            y = k2 * self.position[0] + c2
            x = self.position[0]

        elif k1 < 1e3 and k2 >= 1e3:
            to_return = True
            y = k1 * other.position[0] + c1
            x = other.position[0]

        else:
            to_return = True
            x = (c2 - c1) / (k1 - k2)
            y = k1 * x + c1

        if to_return is not None:
            if other.contains((x, y)):
                if x <= self.position[0]:
                    new_crack = Crack([(int(x), int(y)), self.position])
                    to_return = new_crack + self
                elif x >= self.end_points[1][0]:
                    new_crack = Crack([self.position, (int(x), int(y))])
                    to_return = self + new_crack
                else:
                    to_return = None
            else:
                to_return = None

        return to_return


class CrackNetWork:
    def __init__(self, line_segments, connect=True):
        self.line_segments = list()

        for line_s in line_segments:
            line_list = list()
            for line in line_s:
                if not isinstance(line, Crack):
                    line = Crack(line)
                line_list.append(line)
            line_list = sorted(line_list, key=lambda x: (x[0][0], x[0][1]))
            self.line_segments.append(line_list)
        if connect:
            self.connect_segments()
            self.remove_duplicates()
            self.separate_segments()

            # self.connect(angle_th=np.pi/6, d_th=3, s_th=7, min_crack_length=5, shrt_angle_th=np.pi/3)

    def __str__(self):
        to_return = ""
        for line_s in self.line_segments:
            for line in line_s:
                to_return += str(line) + ", "
            to_return += "\n"
        return to_return

    def __getitem__(self, item):
        return self.line_segments[item]

    def __add__(self, other):
        assert (
            isinstance(other, CrackNetWork) or other is None
        ), "operator + is only defined between two instances of CrackNetWork"
        if other is None:
            return self

        networks = [self, other]
        line_segments = list()
        for nwork in networks:
            for line_s in nwork.line_segments:
                line_segments.append(line_s)
        to_return = CrackNetWork(line_segments, connect=False)
        return to_return

    @property
    def angles(self):
        angle_list = list()

        for line_s in self.line_segments:
            line_angles = list()
            for line in line_s:
                line_angles.append(line.angle)
            angle_list.append(line_angles)

        return angle_list

    @property
    def positions(self):
        position_list = list()
        for line_s in self.line_segments:
            line_list = list()
            for line in line_s:
                line_list.append(line.position)
            position_list.append(line_list)
        return position_list

    @property
    def dims(self):
        min_x = np.inf
        min_y = np.inf
        max_x = 0
        max_y = 0

        for line in self.loop_through_lines():
            l_min_x = min(line.end_points[0][0], line.end_points[1][0])
            l_max_x = max(line.end_points[0][0], line.end_points[1][0])
            l_min_y = min(line.end_points[0][1], line.end_points[1][1])
            l_max_y = max(line.end_points[0][1], line.end_points[1][1])
            if l_min_x < min_x:
                min_x = l_min_x

            min_y = l_min_y if l_min_y < min_y else min_y
            max_x = l_max_x if l_max_x > max_x else max_x
            max_y = l_max_y if l_max_y > max_y else max_y

        return (max_x - min_x), (max_y - min_y)

    def sort(self):
        for line in self.loop_through_lines():
            line.sort()

        for i, line_s in enumerate(self.line_segments):
            self.line_segments[i] = sorted(
                line_s, key=lambda x: (x[0][0], x[1][0], x[0][1], x[1][1])
            )
        pass

    # sorts the line_segments in self so that the next line is a continuation of the previous one
    def sort_by_endpoints(self):
        def correct_id(sorted_ids, id):
            curr_id = 0
            while curr_id <= id:
                if curr_id in sorted_ids:
                    id += 1
                curr_id += 1
            return id

        for j, line_s in enumerate(self.line_segments):
            if len(line_s) <= 1:
                continue
            sorted_ids = list()
            new_positions = list(range(len(line_s)))
            for i in range(len(line_s)):
                _, ids = CrackNetWork.calculate_end_points(
                    [l for (i, l) in enumerate(line_s) if i not in sorted_ids]
                )
                id = correct_id(sorted_ids=sorted_ids, id=ids[0])
                sorted_ids.append(id)
                new_positions[i] = id
            new_line_s = list()
            for val in new_positions:
                new_line_s.append(line_s[val])
            self.line_segments[j] = new_line_s

        return self

    def move_by(self, amount):

        for line_s in self.line_segments:
            for line in line_s:
                line.move_by(amount)
        return self

    def fit_to_mat(self, dims):

        for line_s in self.line_segments:
            for line in line_s:
                line.fit_to_mat(dims)
        return self

    def as_col_img(self, background=None):
        def col_generator(n_lines, n_colours=9):

            for i in range(n_lines):
                j = i % n_colours
                if j < n_colours / 3:
                    g = 3 * j / n_colours
                    r = 1 - g
                    b = 0
                elif j >= 2 * n_colours / 3:
                    b = 3 * (n_colours - j) / n_colours
                    r = 1 - b
                    g = 0

                else:
                    g = (2 * n_colours / 3 - j) * 3 / n_colours
                    b = 1 - g
                    r = 0
                yield (r, g, b)

        def highlight(p, nw, o, col):

            cl_img = np.zeros((o.shape[0], o.shape[1], 3))

            for ls, cl in zip(nw.line_segments, col(len(nw.line_segments))):
                i = p(ls, (o.shape[0], o.shape[1]))
                (cl_red, cl_grn, cl_blue) = cl
                (min_x, min_y), (max_x, max_y) = Crack.bounds(ls)
                for row in range(min_y, max_y + 1):
                    for col in range(min_x, max_x + 1):

                        if (
                            cl_img[row, col, 0] == 0
                            and cl_img[row, col, 1] == 0
                            and cl_img[row, col, 2] == 0
                        ):
                            cl_img[row, col, 0] = (
                                cl_img[row, col, 0] + i[row, col] * cl_red
                            )
                            cl_img[row, col, 1] = (
                                cl_img[row, col, 1] + i[row, col] * cl_grn
                            )
                            cl_img[row, col, 2] = (
                                cl_img[row, col, 2] + i[row, col] * cl_blue
                            )

            for row in range(o.shape[0]):
                for col in range(o.shape[1]):
                    if (
                        cl_img[row, col, 0] == 0
                        and cl_img[row, col, 1] == 0
                        and cl_img[row, col, 2] == 0
                    ):
                        cl_img[row, col, 0] = o[row, col]
                        cl_img[row, col, 1] = o[row, col]
                        cl_img[row, col, 2] = o[row, col]

            return cl_img

        if background is None:
            shape = self.dims
            feed = np.zeros(shape)
        else:
            shape = background.shape
            feed = background

        def mat_gen(ls, shape):
            imgs = np.zeros(shape)
            for line in ls:
                imgs = ip.img_binary_union([imgs, line.as_binmat(shape)])
            return imgs

        nwork_img = highlight(mat_gen, self, feed, col_generator)
        return nwork_img

    # attempts to connect the cracks within each segment so that one crack ends where the next one starts
    def connect_segments(self):

        for i, line_s in enumerate(self.line_segments):
            for j, line1 in enumerate(line_s):
                if j == len(line_s) - 1:
                    break
                min_crack = None
                current_crack = None
                for line2 in line_s[j + 1 :]:
                    if Crack.angle_pass(line1, line2):
                        current_crack = line1.shortest_connecting_line(line2)
                        if current_crack.length > 0:
                            if min_crack is None:
                                min_crack = (
                                    current_crack
                                    if Crack.angle_pass(line1 + current_crack, line2)
                                    else min_crack
                                )
                            elif current_crack.length < min_crack.length:
                                min_crack = (
                                    current_crack
                                    if Crack.angle_pass(line1 + current_crack, line2)
                                    else min_crack
                                )
                if min_crack is not None:
                    self.line_segments[i][j] = line1 + min_crack
                elif current_crack is not None:
                    # endpoints failed, attempt to extrapolate
                    for line2 in line_s[j + 1 :]:
                        extr_line = line1.extrapolate_to(line2)
                        if extr_line is not None:
                            self.line_segments[i][j] = extr_line
                            break
        pass

    # attempts to break down segments by making each crack in them be a continuation of the
    # previous one, removing loops and branches
    def separate_segments(self):

        new_segments = list()
        for line_s in self.line_segments:
            if len(line_s) == 1 and not (line_s[0].length < 1):
                new_segments.append(line_s)
                continue
            seg_groups = list()
            for line in line_s:
                if line.length >= 1:
                    seg_groups.append([line])
            for i in range(len(line_s)):
                matches_done = False
                connections = list()
                for j, grp in enumerate(seg_groups):
                    eps, _ = CrackNetWork.calculate_end_points(grp)
                    for ep in eps:
                        for data in connections:
                            if data[0] == ep:
                                data[1].append(j)
                                break
                        else:
                            connections.append([ep, [j]])
                processed_grps = list()
                for data in connections:
                    grps = data[1]

                    for val in processed_grps:
                        for grp in grps:
                            if grp == val:
                                grps.remove(val)

                    if len(grps) != 2:
                        continue
                    if len(grps) == 2:
                        test_list = list()
                        test_list.extend(seg_groups[grps[0]])
                        test_list.extend(seg_groups[grps[1]])
                        eps, _ = CrackNetWork.calculate_end_points(test_list)
                        if len(eps) == 2:
                            angle_pass = True
                            for line1 in seg_groups[grps[0]]:
                                if angle_pass:
                                    for line2 in seg_groups[grps[1]]:
                                        connector = line1.shortest_connecting_line(
                                            line2
                                        )
                                        if connector.length < 1.0:
                                            if (
                                                line1.angle_difference(
                                                    line2, exact=True
                                                )
                                                > np.pi / 3
                                            ):
                                                angle_pass = False
                                                break
                            if angle_pass:
                                matches_done = True
                                seg_groups[grps[0]].extend(seg_groups[grps[1]])
                                processed_grps.append(grps[1])

                for grp in sorted(processed_grps, reverse=True):
                    seg_groups.pop(grp)
                if not matches_done:
                    break
            for seg in seg_groups:
                new_segments.append(seg)
            self.line_segments = new_segments
        return self

    def loop_through_lines(self):

        for line_s in self.line_segments:
            for line in line_s:
                yield line

    def as_binmat(self, dims):

        mat_list = list()

        for line_s in self.line_segments:
            for line in line_s:
                mat_list.append(line.as_binmat(dims))

        return ip.img_binary_union(mat_list)

    def remove_duplicates(self):

        seg_pop_list = list()
        line_pop_list = list()
        duplicate_list = list()
        for i, line_s in enumerate(self.line_segments):
            for k, line in enumerate(line_s):

                for j, line_s2 in enumerate(self.line_segments):
                    if j < i:
                        continue
                    for l, line2 in enumerate(line_s2):

                        if i == j and k == l:
                            continue
                        if line == line2:
                            duplicate_list.append((i, k, j, l))

        if len(duplicate_list) == 0:
            return self

        for s1, l1, s2, l2 in duplicate_list:
            if len(self.line_segments[s1]) == 1 and s1 not in seg_pop_list:
                seg_pop_list.append(s1)
                continue
            if len(self.line_segments[s2]) == 1 and s2 not in seg_pop_list:
                seg_pop_list.append(s2)
                continue
            if len(self.line_segments[s1]) >= len(self.line_segments[s2]):
                if (s2, l2) not in line_pop_list:
                    line_pop_list.append((s2, l2))
            elif (s1, l1) not in line_pop_list:
                line_pop_list.append((s1, l1))

        if len(seg_pop_list) > 0:
            for i in sorted(seg_pop_list, reverse=True):
                self.line_segments.pop(i)
                if len(line_pop_list) > 0:
                    for val in line_pop_list:
                        if val[0] == i:
                            line_pop_list.remove(val)

        if len(line_pop_list) > 0:
            for i in sorted(line_pop_list, key=lambda x: x[1], reverse=True):
                self.line_segments[i[0]].pop(i[1])
        return self

    # attempts to connect lines within segments to each other
    def connect_line_segments(self, best_line_data):

        line_s1 = self[best_line_data[1]]
        line_s2 = self[best_line_data[2]]

        line1 = line_s1[best_line_data[5]]
        imgs = list()

        for line in line_s1:
            imgs.append(line.as_binmat((256, 256)))
        for line in line_s2:
            imgs.append(line.as_binmat((256, 256)))

        if best_line_data[4]:
            new_mat = (line1 + best_line_data[3]).as_binmat((256, 256))
            self[best_line_data[1]][best_line_data[5]] = line1 + best_line_data[3]
        else:
            new_mat = best_line_data[3].as_binmat((256, 256))
            self[best_line_data[1]].append(best_line_data[3])

        imgs = ip.img_binary_union(imgs)
        ip.plot_array([imgs, new_mat, ip.img_binary_union([imgs, new_mat])])
        self[best_line_data[1]].extend(line_s2)
        self.line_segments.pop(best_line_data[2])

        pass

    # attempts to connect segments to each other
    def connect(
        self,
        angle_th=np.pi / 18,
        d_th=5,
        s_th=32,
        min_crack_length=7,
        shrt_d_th=3,
        shrt_angle_th=np.pi * 5 / 36,
        max_d=256,
    ):
        def rotate_lines(nwork, ind, data):
            connector = data[0]
            line1 = nwork.line_segments[ind][-1 * data[3]]
            line2 = nwork.line_segments[data[2]][-1 * data[4]]

            lines = sorted(
                [line1, line2], key=lambda x: (x[0][0], x[1][0], x[0][1], x[1][1])
            )

            lines[0].end_points[1] = connector.mid_point()
            lines[1].end_points[0] = connector.mid_point()

            if (
                nwork.line_segments[ind][-1 * data[3]][0][0] == lines[0][0][0]
                and nwork.line_segments[ind][-1 * data[3]][0][1] == lines[0][0][1]
            ):
                nwork.line_segments[ind][-1 * data[3]] = lines[0]
                nwork.line_segments[data[2]][-1 * data[4]] = lines[1]
            else:
                nwork.line_segments[ind][-1 * data[3]] = lines[1]
                nwork.line_segments[data[2]][-1 * data[4]] = lines[0]

            nwork.line_segments[ind].extend(nwork.line_segments[data[2]])
            pass

        n_steps = len(self.line_segments)
        for step in range(n_steps):
            changes_made = False
            self.sort()
            best_vals = list()
            # for all line segments figure out the best line segments to connect with
            for i, line_s1 in enumerate(self.line_segments):
                best_vals.append(
                    (None, "val", "other_seg", "first_last", "second_last")
                )
                len_sum = 0
                for line in line_s1:
                    len_sum += line.length

                if len_sum <= min_crack_length:
                    best_vals[i] = (None, None, None, None, None)
                    continue
                end_points1, _ = CrackNetWork.calculate_end_points(line_s1)
                line1 = line_s1[0]
                line2 = None if len(line_s1) == 1 else line_s1[-1]

                min_coords1, max_coords1 = Crack.bounds([line1, line2])

                # decide the best line segment to connect to the current line segment and assign it to best_vals[i]
                # EVERY CHECK MUST BE SYMMETRICAL WITH line_s1!!!
                for j, line_s2 in enumerate(self.line_segments):
                    if i == j:
                        continue
                    line_s2_sum = 0
                    for n_line2 in line_s2:
                        line_s2_sum += n_line2.length

                    if line_s2_sum <= min_crack_length:
                        continue

                    test_list = list()
                    test_list.extend(line_s1)
                    test_list.extend(line_s2)

                    end_points2, _ = CrackNetWork.calculate_end_points(line_s2)

                    # check if line_s1 has an end point that is a point in the middle of one of line_s2 cracks
                    v_nodes = False
                    for ep1 in end_points1:
                        for line in line_s2:
                            for ep2 in line:
                                if ep2 not in end_points2:
                                    if ep1 == ep2:
                                        v_nodes = True
                                        break
                            if v_nodes:
                                break
                        if v_nodes:
                            break

                    if not v_nodes:
                        for ep2 in end_points2:
                            for line in line_s1:
                                for ep1 in line:
                                    if ep1 not in end_points1:
                                        if ep1 == ep2:
                                            v_nodes = True
                                            break
                                if v_nodes:
                                    break
                            if v_nodes:
                                break

                    if v_nodes:
                        continue

                    n_line1 = line_s2[0]
                    n_line2 = None if len(line_s2) == 1 else line_s2[-1]

                    # check if line_s2 is close enough to line_s1 for any connections to be made
                    min_coords2, max_coords2 = Crack.bounds([n_line1, n_line2])
                    min_d = min(
                        map(
                            crack_eukl,
                            [
                                crd1
                                for crd1 in [min_coords1, max_coords1]
                                for crd2 in [min_coords2, max_coords2]
                            ],
                            [
                                crd2
                                for crd2 in [min_coords2, max_coords2]
                                for crd2 in [min_coords2, max_coords2]
                            ],
                        )
                    )
                    if min_d > max_d:
                        continue

                    for k, line in enumerate([line1, line2]):
                        if line is None:
                            continue
                        for l, o_line in enumerate([n_line1, n_line2]):
                            if o_line is None:
                                continue

                            angle_diff = line.angle_difference(o_line)
                            connector = line.shortest_connecting_line(
                                o_line, end_points=(end_points1, end_points2)
                            )
                            if connector is None:
                                continue
                            elif connector.length >= 1.0:
                                test_list.append(connector)
                                eps, _ = CrackNetWork.calculate_end_points(test_list)
                                if len(eps) != 2:
                                    test_list.remove(connector)
                                    continue
                                else:
                                    # check if a coordinate is in line segment more than 2 times,
                                    # indicating a loop or branching
                                    points = list()
                                    loop_check = True
                                    for par in test_list:
                                        if not loop_check:
                                            break
                                        for pep in par.end_points:
                                            points.append(pep)
                                            if len([p for p in points if p == pep]) > 2:
                                                loop_check = False
                                                break
                                    test_list.remove(connector)
                                    if not loop_check:
                                        continue

                            else:
                                eps, _ = CrackNetWork.calculate_end_points(test_list)
                                if len(eps) != 2:
                                    continue
                            if connector.length < 1:
                                angle_diff = line.angle_difference(o_line, exact=True)
                            angl_c = angle_diff < angle_th
                            shrt_angle_c = (connector.length < shrt_d_th) and (
                                angle_diff < shrt_angle_th
                            )
                            pass_check = angl_c or shrt_angle_c

                            if pass_check:
                                # check if the lines are close enough
                                distance_check = connector.length < d_th

                                # check if o_line is in a sector from the end point of line
                                sector1_check = max(
                                    connector.angle_difference(line),
                                    connector.angle_difference(o_line),
                                ) < 2 * angle_th and (connector.length < s_th)

                                # check if o_line is in a wider sector from the end point of line and is heading towards
                                # the middle of the sector
                                sector2_check = (
                                    min(
                                        connector.angle_difference(line),
                                        connector.angle_difference(o_line),
                                    )
                                    < angle_th + np.pi / 36
                                )
                                sector2_check = (
                                    sector2_check and connector.length < s_th
                                )
                                mirror_check = (
                                    connector.angle > line.angle
                                    and o_line.angle <= line.angle
                                ) or (
                                    connector.angle < line.angle
                                    and o_line.angle >= line.angle
                                )
                                mirror_check = mirror_check or (
                                    (
                                        connector.angle > o_line.angle
                                        and line.angle <= o_line.angle
                                    )
                                    or (
                                        connector.angle < o_line.angle
                                        and line.angle >= o_line.angle
                                    )
                                )
                                sector2_check = sector2_check and mirror_check

                                # if the angles are really close to each other, give more slack for the connector's maximum angle
                                angle_mp = 3 if angle_diff <= angle_th / 2 else 1
                                sector3_check = (
                                    max(
                                        connector.angle_difference(line),
                                        connector.angle_difference(o_line),
                                    )
                                    < angle_th * angle_mp
                                    and connector.length < s_th / 2
                                )
                                sector_check = (
                                    sector1_check or sector2_check
                                ) or sector3_check
                                if distance_check or sector_check:
                                    if best_vals[i][0] is None:
                                        changes_made = True
                                        angle_val = np.max(
                                            (
                                                angle_diff,
                                                connector.angle_difference(
                                                    line, exact=True
                                                ),
                                                connector.angle_difference(
                                                    o_line, exact=True
                                                ),
                                            )
                                        )
                                        best_vals[i] = (
                                            connector,
                                            crack_connection_parameter(
                                                angle_val=angle_val,
                                                distance_val=connector.length,
                                            ),
                                            j,
                                            k,
                                            l,
                                        )

                                    else:
                                        angle_val = np.max(
                                            (
                                                angle_diff,
                                                connector.angle_difference(
                                                    line, exact=True
                                                ),
                                                connector.angle_difference(
                                                    o_line, exact=True
                                                ),
                                            )
                                        )
                                        val = crack_connection_parameter(
                                            angle_val=angle_val,
                                            distance_val=connector.length,
                                        )
                                        if best_vals[i][1] > val:
                                            best_vals[i] = (connector, val, j, k, l)

                if best_vals[i][0] is None:
                    best_vals[i] = (None, None, None, None, None)

            # connect two segments to each other if both segments find each other to be the best fit
            if not changes_made:
                break
            connected_list = list()
            for i, data in enumerate(best_vals):
                if data[0] is None or i in connected_list:
                    continue
                d = best_vals[data[2]][2]
                if d == i:
                    if data[0].length >= 0.9:
                        self.line_segments[i].insert(-1 * data[3], data[0])
                    self.line_segments[i].extend(self.line_segments[data[2]])
                    connected_list.append(data[2])
                    self.line_segments[i] = sorted(
                        self.line_segments[i],
                        key=lambda x: (x[0][0], x[1][0], x[0][1], x[1][1]),
                    )

            if len(connected_list) == 0:
                print(
                    "WARNING: matching line segments were found but no matches could be made"
                )
            for i in sorted(connected_list, reverse=True):
                self.line_segments.pop(i)

            self.remove_duplicates()
        self.remove_small()
        pass

    def remove_small(self, th=7):
        pop_list = list()
        for i, line_s in enumerate(self.line_segments):
            total_l = 0
            for l in line_s:
                total_l += l.length
            if total_l < th:
                pop_list.append(i)

        for i in sorted(pop_list, reverse=True):
            self.line_segments.pop(i)
        return self

    # takes in a list of lines and returns the end points of those lists (the points that are only in one of the lines)
    # also returns the index of those lines in the line_seg
    @staticmethod
    def calculate_end_points(line_seg):
        end_point_list = list()
        id_list = list()
        cp_list = list()
        if len(line_seg) == 1:
            end_point_list.append(line_seg[0].end_points[0])
            end_point_list.append(line_seg[0].end_points[1])
            id_list.append(0)
        else:

            for i, line in enumerate(line_seg):
                ep0_conn = False
                ep1_conn = False
                for j, o_line in enumerate(line_seg):
                    if j == i:
                        continue
                    for k, ep1 in enumerate(line.end_points):
                        if ep1 not in end_point_list:
                            end_point_list.append(ep1)
                        for ep2 in o_line.end_points:
                            if ep1[0] == ep2[0] and ep1[1] == ep2[1]:

                                if ep1 not in cp_list:
                                    cp_list.append(ep1)
                                if k == 0:
                                    ep0_conn = True
                                else:
                                    ep1_conn = True
                if not (ep0_conn and ep1_conn):
                    id_list.append(i)

            if len(cp_list) > 0:
                for p in cp_list:
                    end_point_list.remove(p)
        return end_point_list, id_list

    def v_nodes(self, dims, th=4, ext_l=10):

        v_list = list()
        # create a list of end_points and ids for every segment pair
        for i, line_s1 in enumerate(self.line_segments):

            end_points1, id_list1 = CrackNetWork.calculate_end_points(line_s1)

            seg1_len = sum([l.length for l in line_s1])
            for j, line_s2 in enumerate(self.line_segments):
                if j <= i:
                    continue
                end_points2, id_list2 = CrackNetWork.calculate_end_points(line_s2)
                seg2_len = sum([l.length for l in line_s2])
                # if the end points are close enough, add to the v_list
                for ep1 in end_points1:
                    for ep2 in end_points2:
                        if crack_eukl(ep1, ep2) < th:
                            if seg1_len >= seg2_len:
                                v_list.append((i, ep1, id_list1))
                            else:
                                v_list.append((j, ep2, id_list2))
        if len(v_list) > 0:
            for i, p, l_list in v_list:
                for k, line in enumerate(self.line_segments[i]):
                    if k not in l_list:
                        continue
                    else:
                        for s, ep in enumerate(line.end_points):
                            if p[0] == ep[0] and p[1] == ep[1]:
                                self.line_segments[i][k] = line.extend_by(
                                    ext_l, side=s, dims=dims
                                )
        return self

    # attempts to combine all the nworks in nwork_list corresponding to their position
    @staticmethod
    def combine_nworks(nwork_list, shape, n_mats_per_row):
        assert (
            len(nwork_list) % n_mats_per_row == 0
        ), f"len of nwork_list must be a multiple of {n_mats_per_row}"
        w, h = shape

        curr_row = -1

        nworks = CrackNetWork([], connect=False)
        for i, nwork in enumerate(nwork_list):
            curr_col = i % n_mats_per_row
            if curr_col == 0:
                curr_row += 1
            if i == 0 or nwork is None:
                continue
            step = (w * curr_col, h * curr_row)
            nwork.move_by(step)
            nworks = nworks + nwork

        nworks.connect(s_th=64, min_crack_length=32)
        nworks = nworks.sort_by_endpoints()
        nworks = nworks.v_nodes(
            dims=(int(len(nwork_list) / n_mats_per_row) * h, int(n_mats_per_row * w))
        )
        return nworks

    def to_geodataframe(self, orig_shape, orig_bounds):
        def find_path(line_s, conn_list):
            seg_list = list()
            working = True
            while (len(conn_list) < len(line_s)) and working:
                working = False
                for i, line in enumerate(line_s):
                    if i in conn_list:
                        continue
                    for s, ep in enumerate(line.end_points):
                        if len(seg_list) == 0:
                            eps, _ = CrackNetWork.calculate_end_points(line_s)
                            # TODO: Understand what all of this does
                            if len(eps) == 0 or len(line.end_points) == 0:
                                # TODO: Just added to avoid IndexError
                                conn_list.append(i)
                                break
                            # One might be empty for some reason in the if
                            # eps[]... statement
                            # IndexError: list index out of range
                            if eps[0] == line.end_points[0]:
                                seg_list.extend(
                                    [line.end_points[0], line.end_points[1]]
                                )
                            else:
                                seg_list.extend(
                                    [line.end_points[1], line.end_points[0]]
                                )
                            conn_list.append(i)
                            break
                        elif ep == seg_list[-1]:
                            seg_list.append(line.end_points[s - 1])
                            conn_list.append(i)
                            working = True
                            break
            return seg_list, conn_list

        df = pd.DataFrame({"ID": [], "Lines": []})
        curr_ID = 1
        convX = linear_converter(
            [0, orig_shape[1] - 1], [orig_bounds[0], orig_bounds[2]], ignore_errors=True
        )
        convY = linear_converter(
            [orig_shape[0] - 1, 0], [orig_bounds[1], orig_bounds[3]], ignore_errors=True
        )

        for line_s in self.line_segments:

            conn_list = list()
            add_seg_lists = list()

            while len(conn_list) < len(line_s):
                seg_list, conn_list = find_path(line_s, conn_list)
                add_seg_lists.append(seg_list)
            for seg in add_seg_lists:
                conv_valsX = map(convX, [x[0] for x in seg])
                conv_valsY = map(convY, [x[1] for x in seg])
                conv_valsX = [x for x in conv_valsX]
                conv_valsY = [y for y in conv_valsY]
                vals = [Point(xy) for xy in zip(conv_valsX, conv_valsY)]

                df = df.append({"ID": curr_ID, "Lines": vals}, ignore_index=True)
                curr_ID += 1
        df = df["Lines"].apply(lambda x: LineString(x))
        gdf = geopandas.GeoDataFrame(df, geometry="Lines")
        return gdf
