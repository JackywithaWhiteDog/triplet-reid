#!bin/bash

DEVICE=0

IMAGE_ROOT=/home/jacky/Documents/triplet-reid
EXP_ROOT=/home/jacky/Documents/triplet-reid/experiments/finetune

rm -r $EXP_ROOT

CUDA_VISIBLE_DEVICES=$DEVICE \
    python train.py \
        --train_set data/market1501_train.csv \
        --model_name resnet_v1_50 \
        --image_root $IMAGE_ROOT \
        --experiment_root $EXP_ROOT \
        --initial_checkpoint ./resnet_v1_50.ckpt \
        --flip_augment \
        --crop_augment \
        --embedding_dim 128 \
        --batch_p 18 \
        --batch_k 4 \
        --pre_crop_height 288 --pre_crop_width 144 \
        --net_input_height 256 --net_input_width 128 \
        --margin soft \
        --metric euclidean \
        --loss batch_hard \
        --learning_rate 3e-4 \
        --train_iterations 25000 \
        --decay_start_iteration 15000
