"""
Tests for crack_main.py.
"""
import os
from contextlib import nullcontext
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString

import tests
from alsa import crack_cls, crack_main, utils


@pytest.mark.parametrize(
    "test_image,raises",
    [
        pytest.param(
            tests.KL5_TEST_IMAGE, nullcontext(), id="8-bit-png-prediction-success"
        ),
        pytest.param(
            tests.KL5_TEST_IMAGE_16BIT,
            pytest.raises(utils.TooHomogeneousImageError),
            id="16-bit-prediction-fail",
        ),
    ],
)
@pytest.mark.skipif(
    os.environ.get("CI") is not None, reason="Tensorflow crashes on Github Actions."
)
def test_crack_main(test_image, raises, tmp_path):
    """
    Test crack_main.
    """

    def _check_segmenting(tmp_path):
        assert len(list((tmp_path / "sub_imgs").iterdir())) > 0

    traces_path = tmp_path / "test_crack_main_traces.shp"
    with raises:
        try:
            combined_nwork, orig_dims, geo_data, result_gdf = crack_main.crack_main(
                work_dir=tmp_path,
                img_path=test_image,
                area_file_path=list(tests.KL5_TEST_AREA_DIR.glob("*.shp"))[0],
                unet_weights_path=tests.KL5_TEST_WEIGHTS,
                predicted_output_path=traces_path,
            )
        except FileNotFoundError:
            if not tests.KL5_TEST_WEIGHTS.exists():
                # Check that segmenting works even if model is missing
                _check_segmenting(tmp_path=tmp_path)
                pytest.xfail(
                    "Skipping prediction in test_crack_main as weights are missing."
                )
            raise

        _check_segmenting(tmp_path=tmp_path)

        assert isinstance(combined_nwork, crack_cls.CrackNetWork)
        assert isinstance(geo_data, gpd.GeoDataFrame)
        assert traces_path.exists()
        assert isinstance(result_gdf, gpd.GeoDataFrame)
        assert len(orig_dims) == 2

        gdf = gpd.read_file(traces_path)

        assert gdf.shape[0] > 0
        assert all(
            isinstance(geom, (LineString, MultiLineString))
            for geom in gdf.geometry.values
        )


@pytest.mark.parametrize(
    "override_ridge_config_path", tests.test_resolve_ridge_config_overrides_params()
)
def test_resolve_ridge_config_overrides(override_ridge_config_path, tmp_path):
    """
    Test resolve_ridge_config_overrides.
    """
    result = crack_main.resolve_ridge_config_overrides(
        override_ridge_config_path=override_ridge_config_path,
        work_dir=tmp_path,
    )

    assert isinstance(result, dict)

    if override_ridge_config_path is None:
        assert len(result) == 0
    else:
        assert len(result) > 0


def test_resolve_ridge_config_overrides_default(tmp_path: Path):
    """
    Test resolve_ridge_config_overrides.
    """
    override_ridge_config_path = tmp_path / tests.SAMPLE_RIDGE_CONFIG_PATH.name
    override_ridge_config_path.write_text(tests.SAMPLE_RIDGE_CONFIG_PATH.read_text())
    result = crack_main.resolve_ridge_config_overrides(
        override_ridge_config_path=None,
        work_dir=tmp_path,
    )

    assert isinstance(result, dict)
    assert len(result) > 0
