#!/bin/bash

python CRAWFORD_train.py \
--save_dir /path/to/save_directory \
--train_data_dir /path/to/training_directory \
--num_epochs 10 \
--batch_size 64 \
--img_size 256 \
--unet_number_to_train 2 \
--num_samples_save 600