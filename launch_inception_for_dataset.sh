#!/bin/bash
#'/home/ubuntu/kinetics-400/kinetics/Kinetics_trimmed_videos_train_merge'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/bin/python3 calculate_inception_moments.py \
--dataset Kinetics400 --parallel --shuffle \
--num_workers 32 --batch_size 96 \
--time_steps 12 \
--frames_between_clips 1000000 \
--seed 0 \
--data_root '/home/ubuntu/kinetics-400/kinetics/Kinetics_trimmed_videos_train_merge' \
--logs_root '/home/ubuntu/nfs/xdu12/dvd-gan/logs/' \
