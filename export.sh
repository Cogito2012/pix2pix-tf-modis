#!/bin/bash

python pix2pix.py \
      --mode export \
      --output_filetype png \
      --crop_size 512 \
      --scale_size 542 \
      --output_dir output/modcn_jun_export \
      --checkpoint ./output/modcn_jun_train


python ./server/tools/process-local.py \
        --model_dir output/modcn_jun_export \
        --input_file 1_A.png \
        --output_file test_export.png
