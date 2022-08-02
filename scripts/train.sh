#!/usr/bin/env bash

poetry run python -m alsa train ../ALSA-loviisa/ \
    --epochs 10 \
    --validation-steps 5 \
    --steps-per-epoch 5 \
    --trace-width 0.015 \
    --batch-size 32
