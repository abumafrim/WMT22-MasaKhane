model="albert-large-v2"

for lang in eng-hau eng-ibo eng-lug eng-swh eng-tsn eng-yor eng-zul fra-wol; do
    train=data/$lang/spc-$lang"_train.tsv"
    dev=data/$lang/spc-$lang"_dev.tsv"
    test=data/$lang/spc-$lang"_test.tsv"

    model_path=models/$model/$lang
    mkdir -p $model_path

    python scripts/run-sp-class.py \
        --train \
        --eval=True \
        --model=$model \
        --model_path=$model_path \
        --train_data=$train \
        --val_data=$dev \
        --test_data=$test \
        --epochs=4
done