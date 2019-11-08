#!/bin/bash
#xiaodan: add time step and k here
# xiaodan: delete --hier
#--G_attn 16 --D_attn 16 \
#--ema --use_ema --ema_start 20000 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/bin/python3 train.py \
--dataset UCF101_32 --annotation_file '/home/nfs/data/trainlist01.txt' --parallel --shuffle \
--num_workers 8 --batch_size 48 --load_in_mem  \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 5000 \
--num_D_steps 1 --G_lr 2e-4 --D_lr 2e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 32 --D_attn 0 \
--time_steps 12 \
--k 8 --frames_between_clips 1000000 \
--G_nl relu --D_nl relu \
--SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
--G_ortho 0.0 \
--frame_size 32 \
--G_init N02 --D_init N02 \
--dim_z 128 --G_shared --shared_dim 128 \
--G_ch 64 --D_ch 64 \
--ema --use_ema --ema_start 1000 \
--test_every 100 --save_every 100 --num_best_copies 5 --num_save_copies 0 --seed 0 \
--data_root ../../data/UCF-101_copy2 \
--logs_root '/home/ubuntu/nfs/xdu12/dvd-gan/logs/'
#--G_mixed_precision --D_mixed_precision \
# --which_train_fn dummy
