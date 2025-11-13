#!/bin/bash

python CRAWFORD_generate.py \
--h5_path /path/to/h5_file.h5 \
--checkpoint /path/to/model.pt \
--output_dir /path/to/output_directory \
--num_imgs_per_emb 5
