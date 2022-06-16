export MODEL=albert-base-v2
export TRAIN_DATA=data/en-hau/spc-en_hau_train.tsv
export VAL_DATA=data/en-hau/spc-en_hau_dev.tsv
export TEST_DATA=data/en-hau/spc-en_hau_test.tsv
export NUM_EPOCHS=4

python3 run-sp-class.py \
    --train \
    --eval=True \
    --model=$MODEL \
    --train_data=$TRAIN_DATA \
    --val_data=$VAL_DATA \
    --test_data=$TEST_DATA \
    --epochs=$NUM_EPOCHS