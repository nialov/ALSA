"""
General utilities.
"""
import logging

import numpy as np


class TooHomogeneousImageError(Exception):
    """
    Input image is too homogeneous.
    """


def report_redundant_proportion(redundant_proportion: float):
    """
    Report or raise an error if input image is too homogeneous.
    """
    redundant_proportion_perc = redundant_proportion * 100
    redundant_text = f"The number of homogeneous images is {redundant_proportion_perc} % of all images"
    if redundant_proportion > 0.9:
        logging.error(redundant_text)
    else:
        logging.info(redundant_text)
    if np.isclose(redundant_proportion, 1.0):
        raise TooHomogeneousImageError(
            f"The proportion of homogeneous images is {redundant_proportion_perc} %."
            " Check that your input images to prediction are 8-bit pngs."
        )


def conditional_print(text: str, is_verbose: bool):
    if is_verbose:
        print(text)
