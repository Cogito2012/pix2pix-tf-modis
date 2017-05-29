#!/bin/bash

python translate_multiple_image.py \
        --model_dir output/modcn_jun_export2 \
        --device_opts /cpu:0 \
        --input_dir ./testdata/patches2 \
        --output_dir ./testdata/patches_translate2