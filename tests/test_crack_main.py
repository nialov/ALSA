"""
Tests for crack_main.py.
"""
from warnings import warn

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString

import tests
from alsa import crack_main


def test_crack_main(tmp_path):
    """
    Test crack_main.
    """
    traces_path = tmp_path / "test_crack_main_traces.shp"
    if not tests.KL5_TEST_WEIGHTS.exists():
        warn("Skipping test_crack_main as weights are missing.")
        return
    nworks, orig_dims, geo_data, result_gdf = crack_main.crack_main(
        work_dir=tmp_path,
        img_path=tests.KL5_TEST_IMAGE,
        area_shp_file_path=list(tests.KL5_TEST_AREA_DIR.glob("*.shp"))[0],
        unet_weights_path=tests.KL5_TEST_WEIGHTS,
        new_shp_path=traces_path,
    )
    assert traces_path.exists()
    assert isinstance(result_gdf, gpd.GeoDataFrame)
    assert len(orig_dims) == 2

    gdf = gpd.read_file(traces_path)

    assert gdf.shape[0] > 0
    assert all(
        isinstance(geom, (LineString, MultiLineString)) for geom in gdf.geometry.values
    )