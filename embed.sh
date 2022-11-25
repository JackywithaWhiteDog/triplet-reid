#!bin/bash

DEVICE=7

IMAGE_ROOT=/home/jacky/Documents/triplet-reid
EXP_ROOT=/home/jacky/Documents/triplet-reid/experiments/finetune

for EPOCH in 0 5000 10000 15000 20000 25000
do
    CUDA_VISIBLE_DEVICES=$DEVICE \
        python embed.py \
            --experiment_root $EXP_ROOT \
            --checkpoint checkpoint-$EPOCH \
            --dataset data/market1501_train.csv \
            --filename market1501_train_embeddings_$EPOCH.h5 \
            --flip_augment \
            --crop_augment five \
            --aggregator mean
    CUDA_VISIBLE_DEVICES=$DEVICE \
        python embed.py \
            --experiment_root $EXP_ROOT \
            --checkpoint checkpoint-$EPOCH \
            --dataset data/market1501_test.csv \
            --filename market1501_test_embeddings_$EPOCH.h5 \
            --flip_augment \
            --crop_augment five \
            --aggregator mean
    CUDA_VISIBLE_DEVICES=$DEVICE \
        python embed.py \
            --experiment_root $EXP_ROOT \
            --checkpoint checkpoint-$EPOCH \
            --dataset data/market1501_query.csv \
            --filename market1501_query_embeddings_$EPOCH.h5 \
            --flip_augment \
            --crop_augment five \
            --aggregator mean
done

# CUDA_VISIBLE_DEVICES=$DEVICE \
#     python embed.py \
#         --experiment_root $EXP_ROOT \
#         --checkpoint checkpoint-25000 \
#         --dataset data/market1501_train.csv \
#         --filename market1501_train_embeddings.h5 \
#         --crop_augment five \
#         --aggregator mean

# CUDA_VISIBLE_DEVICES=$DEVICE \
#     python embed.py \
#         --experiment_root $EXP_ROOT \
#         --checkpoint checkpoint-25000 \
#         --dataset data/market1501_query.csv \
#         --filename market1501_query_embeddings.h5 \
#         --crop_augment five \
#         --aggregator mean

# CUDA_VISIBLE_DEVICES=$DEVICE \
#     python embed.py \
#         --experiment_root $EXP_ROOT \
#         --checkpoint checkpoint-25000 \
#         --dataset data/market1501_test.csv \
#         --filename market1501_test_embeddings.h5 \
#         --crop_augment five \
#         --aggregator mean
