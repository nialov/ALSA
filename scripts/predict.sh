#!/usr/bin/env bash

poetry run python -m alsa predict ../ALSA-loviisa/ \
        --img-path tests/sample_data/kl5_test_data/kl5_subsample.png \
        --area-file-path tests/sample_data/kl5_test_data/area/kl5_subsample_bounds.shp \
        --unet-weights-path ../ALSA-loviisa/unet_weights.hdf5 \
        --driver GeoJSON \
        --predicted-output-path ../ALSA-loviisa/kl5_sub.geojson
