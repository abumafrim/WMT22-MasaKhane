export MAX_LENGTH=128
export MODEL=albert-base-v2
export OUTPUT_DIR=pt2_mbert
export TRAIN_DATA=data/en-hau/spc-en_hau_train.tsv
export VAL_DATA=data/en-hau/spc-en_hau_dev.tsv
export TEST_DATA=data/en-hau/spc-en_hau_test.tsv
export BATCH_SIZE=32
export NUM_EPOCHS=2
export LEARNING_RATE=2e-5
export FREEZE_BERT=False
export ITERS_TO_ACCUMULATE=2
export PRED_THRESHOLD=0.5
export SEED=2

python3 run-sp-class.py \
    --train \
    --eval=True \
    --model=$MODEL \
    --epochs=$NUM_EPOCHS \
    --train_data=$TRAIN_DATA \
    --val_data=$VAL_DATA \
    --test_data=$TEST_DATA \
    --freeze_bert=$FREEZE_BERT \
    --maxlen=$MAX_LENGTH \
    --batch_size=$BATCH_SIZE \
    --learning_rate=$LEARNING_RATE \
    --iters_to_accumulate=$ITERS_TO_ACCUMULATE \
    --pred_threshold=$PRED_THRESHOLD \
    --seed=$SEED