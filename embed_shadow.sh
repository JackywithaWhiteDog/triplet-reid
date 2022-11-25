#!bin/bash

DEVICE=6

IMAGE_ROOT=/home/jacky/Documents/triplet-reid
EXP_ROOT=/home/jacky/Documents/triplet-reid/experiments/shadow
DOMAIN=shadow

EPOCH=25000

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
