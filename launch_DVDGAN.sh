#!/bin/bash
#xiaodan: add time step and k here
# xiaodan: delete --hier
#--G_attn 16 --D_attn 16 \
#--ema --use_ema --ema_start 20000 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/bin/python3 train.py \
--dataset UCF101 --annotation_file '/home/nfs/data/trainlist01.txt' --parallel --shuffle \
--num_workers 8 --batch_size 16 --load_in_mem  \
--num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 0 \
--time_steps 12 \
--k 8 --frames_between_clips 1000000 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared --frame_size 64 \
--G_init ortho --D_init ortho \
--dim_z 120 --shared_dim 120 \
--G_ch 96 --D_ch 96 \
--ema --use_ema --ema_start 20000 \
--test_every 100 --save_every 100 --num_best_copies 5 --num_save_copies 0 --seed 0 \
--data_root ../../data/UCF-101_copy2 \
--use_multiepoch_sampler \
--logs_root '/home/ubuntu/nfs/xdu12/dvd-gan/logs/'
#--G_mixed_precision --D_mixed_precision \
# --which_train_fn dummy
