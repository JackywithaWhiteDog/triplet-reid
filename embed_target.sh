#!bin/bash

DEVICE=7

IMAGE_ROOT=/home/jacky/Documents/triplet-reid
EXP_ROOT=/home/jacky/Documents/triplet-reid/experiments/target
DOMAIN=ground_truth

for EPOCH in 0 5000 10000 15000 20000 25000
do
    for SPLIT in training_member non_training_member non_member
    do
        CUDA_VISIBLE_DEVICES=$DEVICE \
            python embed.py \
                --experiment_root $EXP_ROOT \
                --checkpoint checkpoint-$EPOCH \
                --dataset splitted_data/${DOMAIN}_${SPLIT}.csv \
                --filename market1501_${DOMAIN}_${SPLIT}_embeddings_$EPOCH.h5 \
                --flip_augment \
                --crop_augment five \
                --aggregator mean
    done
done
