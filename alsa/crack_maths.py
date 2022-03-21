"""
Mathematical utilities.
"""
import numpy as np


def linear_converter(orig_range, new_range, ignore_errors=False):
    """
    Return a function that converts points within orig_range to points in new_range.
    """
    assert (
        len(orig_range) == 2
    ), f"argument orig_range must be a list-like with len 2. Was given {len(orig_range)}"
    assert (
        len(new_range) == 2
    ), f"argument orig_range must be a list-like with len 2. Was given {len(new_range)}"

    if orig_range[0] == orig_range[1]:
        raise ZeroDivisionError(
            f"linear_converter was given orig_range of {orig_range}"
        )
    if new_range[0] == new_range[1]:
        raise ZeroDivisionError(f"linear_converter was given new_range of {new_range}")

    def lin_func(x):
        if not ignore_errors:
            assert (
                orig_range[0] <= x <= orig_range[1]
                or orig_range[0] >= x >= orig_range[1]
            ), f"{x} is out of bounds of {orig_range}"

        x_d = x - orig_range[0]
        d = orig_range[1] - orig_range[0]
        new_d = new_range[1] - new_range[0]

        return new_d * (x_d) / d + new_range[0]

    return lin_func


def line_point_generator(crd1, crd2):
    """
    Generate points along line.

    >>> list(line_point_generator(np.array([0, 0]), np.array([4, 4])))
    [(0, 0), (1, 1), (1, 1), (2, 2), (2, 2), (3, 3), (3, 3), (4, 4)]
    """
    x_d = abs(crd2[0] - crd1[0])
    y_d = abs(crd2[1] - crd1[1])

    n_steps = 2 * np.max((x_d, y_d))

    for x, y in zip(
        np.linspace(crd1[0], crd2[0], n_steps), np.linspace(crd1[1], crd2[1], n_steps)
    ):
        yield int(round(x)), int(round(y))
