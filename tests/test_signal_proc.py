"""
Test alsa.signal_proc.
"""

import pytest

from alsa import signal_proc


@pytest.mark.parametrize(
    "img_path,override_ridge_configs",
    [
        ("image.png", {}),
        (
            "image.png",
            {
                "optional_parameters": {"Line_width": 2},
                "further_options": {"Correct_position": False},
            },
        ),
    ],
)
def test_resolve_ridge_config(img_path, override_ridge_configs: dict):
    """
    Test resolve_ridge_config.
    """
    result = signal_proc.resolve_ridge_config(
        img_path=img_path, override_ridge_configs=override_ridge_configs
    )

    assert isinstance(result, dict)
    assert result["path_to_file"] == img_path

    for key, value in override_ridge_configs.items():
        assert key in result
        assert type(value) == type(result[key])
        if isinstance(result[key], dict) and isinstance(value, dict):
            for sub_key, sub_value in value.items():
                assert sub_key in result[key]
                assert sub_value == result[key][sub_key]
