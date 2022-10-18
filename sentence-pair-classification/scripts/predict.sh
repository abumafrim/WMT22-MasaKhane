auto_aligned=('wmt22_african' 'lava-corpus' 'webcrawl_african' 'WikiMatrix' 'CCAligned' 'CCMatrix' 'ParaCrawl' 'GNOME' 'KDE4' 'TED2020' 'XLEnt' 'Ubuntu' 'wikimedia' 'MultiCCAligned')

model=albert-xlarge-v2
basepath=../../data/filtering
modelspath=../models
#very large loss (arbitrary)
valloss=1000

filename=$basepath/$model'-done_pred.txt'
IFS=$'\r\n' GLOBIGNORE='*' command eval  'donepred=($(cat filename))'
echo "$filename"
echo ${donepred[@]}