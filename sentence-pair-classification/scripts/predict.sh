auto_aligned=('wmt22_african' 'lava-corpus' 'webcrawl_african' 'WikiMatrix' 'CCAligned' 'CCMatrix' 'ParaCrawl' 'GNOME' 'KDE4' 'TED2020' 'XLEnt' 'Ubuntu' 'wikimedia' 'MultiCCAligned')

model=albert-xlarge-v2
base_path=../../data/filtering
models_path=../models
#very large loss (arbitrary)
val_loss=1000

filename=$base_path/$model'-done_pred.txt'
IFS=$'\r\n' GLOBIGNORE='*' command eval  'done_pred=($(cat filename))'

echo ${done_pred[@]}