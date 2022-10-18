model=albert-xlarge-v2
basepath=../../data/filtering
modelspath=../models
#very large loss (arbitrary)
val_loss=2.0

donefile=$basepath/$model"-done_pred.txt"
if [ -f $donefile ]; then
    readarray -t donepred < $donefile
fi

value=webcrawl_african'\t'eng-hau

if [[ ! " ${donepred[*]} " =~ " ${value} " ]]; then
    echo "$value"
    echo "Not found"
fi