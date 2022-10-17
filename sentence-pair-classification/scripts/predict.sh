export MODEL=albert-base-v2
export MODEL_PATH=models/albert-base-v2_lr_2e-05_val_loss_0.63068_ep_4.pt
export DATA_PATH=data/en-hau/spc-en_hau_10000_to_pred.tsv
export OUTPUT_PATH=results/
export PRED_THRESHOLD=0.5

python3 predict.py \
    --predict \
    --model=$MODEL \
    --model_path=$MODEL_PATH \
    --data_path=$DATA_PATH \
    --output_path=$OUTPUT_PATH \
    --pred_threshold=$PRED_THRESHOLD \