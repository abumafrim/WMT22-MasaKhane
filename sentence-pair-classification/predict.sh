export MAX_LENGTH=128
export MODEL=albert-base-v2
export MODEL_PATH=models/albert-base-v2_lr_2e-05_val_loss_0.15311_ep_1.pt
export DATA_PATH=data/en-hau/spc-en_hau_10000_to_pred.tsv
export OUTPUT_PATH=results/
export BATCH_SIZE=32
export PRED_THRESHOLD=0.5
export SEED=2

python3 predict.py \
    --predict \
    --model_path=$MODEL_PATH \
    --data_path=$DATA_PATH \
    --output_path=$OUTPUT_PATH \
    --maxlen=$MAX_LENGTH \
    --batch_size=$BATCH_SIZE \
    --pred_threshold=$PRED_THRESHOLD \
    --seed=$SEED