#!/bin/bash

python pix2pix.py \
      --mode train \
      --output_dir output/modcn_jun_train2 \
      --batch_size 8 \
      --max_epochs 200 \
      --crop_size 512 \
      --scale_size 542 \
      --input_dir ./datasets/modcn_jun/train \
      --which_direction AtoB