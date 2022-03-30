"""
Tests for model.py.
"""
import pytest
from tensorflow.keras.models import Model

import tests
from alsa import model


@pytest.mark.parametrize(
    "pretrained_weigths,input_size",
    [
        (None, (256, 256, 1)),
        (tests.KL5_TEST_WEIGHTS, (256, 256, 1)),
        (None, (128, 128, 1)),
    ],
)
def test_unet(pretrained_weigths, input_size):
    """
    Test unet.
    """
    if pretrained_weigths is not None and not pretrained_weigths.exists():
        pytest.xfail(
            "Model weights were not found on disk. Skipping current test_unet test."
        )
    result = model.unet(pretrained_weights=pretrained_weigths, input_size=input_size)

    assert isinstance(result, Model)
