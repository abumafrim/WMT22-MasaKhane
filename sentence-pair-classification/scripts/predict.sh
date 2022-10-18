model=albert-xlarge-v2
basepath=../../data/filtering
modelspath=../models
#very large loss (arbitrary)
valloss=1000

donefile=$basepath/$model"-done_pred.txt"
if [ -f $donefile ]; then
    readarray -t donepred < $donefile
fi

for data in wmt22_african lava-corpus webcrawl_african WikiMatrix CCAligned CCMatrix ParaCrawl GNOME KDE4 TED2020 XLEnt Ubuntu wikimedia MultiCCAligned; do
  datapath=$basepath/$data
  for sfile in $datapath/*.tsv; do
    value=$data$'\t'$sfile
    if [[ ! " ${donepred[*]} " =~ " ${value} " ]]; then
        lang="$(cut -d'/' -f1 <<<"$sfile")"
        lang=${lang//.tsv}
        echo "Predicting the quality of: $lang"

    elif [[ " ${donepred[*]} " =~ " ${value} " ]]; then
	    echo "$value finished."
    fi
  done
  echo ''
done