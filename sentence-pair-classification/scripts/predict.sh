model=albert-xlarge-v2
basepath=../../data/filtering
modelspath=../models
#very large loss (arbitrary)
val_loss=2.0

donefile=$basepath/$model"-done_pred.txt"
if [ -f $donefile ]; then
    readarray -t donepred < $donefile
fi

for data in wmt22_african lava-corpus webcrawl_african WikiMatrix CCAligned CCMatrix ParaCrawl GNOME KDE4 TED2020 XLEnt Ubuntu wikimedia MultiCCAligned; do
  datapath=$basepath/$data
  for sfile in $datapath/*.tsv; do
    lang="$(cut -d'/' -f6 <<<"$sfile")"
    lang=${lang//.tsv}
    value=$data$' '$lang
    if [[ ! " ${donepred[*]} " =~ " ${value} " ]]; then
        model_path=$modelspath/$model/$lang
        for x in $model_path/$model*.pt; do
            loss="$(cut -d'_' -f6 <<<"$x")"
            if awk "BEGIN {exit !($loss < $val_loss)}"; then
                model_path=$x
            else
                model_path=""
            fi
        done

        data_to_classify=$sfile
        save_to=$datapath/$model"_"$lang".preds"

        echo "Finished: $data $lang"

        echo $value >> $donefile

    elif [[ " ${donepred[*]} " =~ " ${value} " ]]; then
	    echo "Finished: $data $lang already"
    fi
  done
  echo ''
done