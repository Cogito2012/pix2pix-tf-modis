#!/bin/bash

python pix2pix.py \
      --mode test \
      --output_dir output/modcn_jun_test2 \
      --input_dir ./datasets/modcn_jun/test \
      --crop_size 512 \
      --scale_size 542 \
      --checkpoint ./output/modcn_jun_train2