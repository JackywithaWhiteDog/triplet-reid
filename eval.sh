#!bin/bash

DEVICE=6

IMAGE_ROOT=/home/jacky/Documents/triplet-reid
EXP_ROOT=/home/jacky/Documents/triplet-reid/experiments/finetune

EPOCH=0
CUDA_VISIBLE_DEVICES=$DEVICE \
    python ./evaluate.py \
        --excluder market1501 \
        --query_dataset data/market1501_query.csv \
        --query_embeddings $EXP_ROOT/market1501_query_embeddings_$EPOCH.h5 \
        --gallery_dataset data/market1501_test.csv \
        --gallery_embeddings $EXP_ROOT/market1501_test_embeddings_$EPOCH.h5 \
        --metric euclidean \
        --filename $EXP_ROOT/market1501_evaluation_$EPOCH.json \
        --use_market_ap

# CUDA_VISIBLE_DEVICES=$DEVICE \
#     python ./evaluate.py \
#     --excluder market1501 \
#         --query_dataset data/market1501_query.csv \
#         --query_embeddings $EXP_ROOT/market1501_query_embeddings.h5 \
#         --gallery_dataset data/market1501_test.csv \
#         --gallery_embeddings $EXP_ROOT/market1501_test_embeddings.h5 \
#         --metric euclidean \
#         --filename $EXP_ROOT/market1501_evaluation.json \
        # --use_market_ap

# Pretrained
# mAP: 68.98% | top-1: 84.20% top-2: 89.61% | top-5: 94.36% | top-10: 96.14%

# Sample
# mAP: 7.01% | top-1: 16.09% top-2: 23.34% | top-5: 34.71% | top-10: 44.80%

# Sample 2
# mAP: 3.22% | top-1: 7.72% top-2: 11.43% | top-5: 18.71% | top-10: 25.86%

# resnet_v2_50
# mAP: 9.17% | top-1: 19.27% top-2: 26.54% | top-5: 39.58% | top-10: 50.89%

# lunet_50
# mAP: 5.83% | top-1: 12.95% top-2: 18.91% | top-5: 28.44% | top-10: 38.48%

# Fine-tune
# mAP: 64.18% | top-1: 81.77% top-2: 87.23% | top-5: 92.52% | top-10: 94.86%

# Fine-tune 0
# mAP: 2.52% | top-1: 10.10% top-2: 13.90% | top-5: 20.61% | top-10: 26.69%

# Fine-tune 5000
# mAP: 58.78% | top-1: 79.16% top-2: 85.45% | top-5: 90.65% | top-10: 94.00%

# Fine-tune 10000
# mAP: 58.34% | top-1: 78.92% top-2: 85.15% | top-5: 91.03% | top-10: 94.36%

# Fine-tune 15000
# mAP: 60.29% | top-1: 79.60% top-2: 85.75% | top-5: 92.01% | top-10: 95.13%

# Fine-tune 20000
# mAP: 68.88% | top-1: 85.33% top-2: 89.79% | top-5: 94.39% | top-10: 96.62%

# Fine-tune 25000
# mAP: 70.11% | top-1: 85.39% top-2: 89.99% | top-5: 94.39% | top-10: 96.59%
