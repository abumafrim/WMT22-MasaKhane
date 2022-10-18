model=albert-xlarge-v2
basepath=../../data/filtering
modelspath=../models
#very large loss (arbitrary)
val_loss=1000

donefile=$basepath/$model"-done_pred.txt"
if [ -f $donefile ]; then
    readarray -t donepred < $donefile
fi

for data in wmt22_african lava-corpus webcrawl_african WikiMatrix CCAligned CCMatrix ParaCrawl GNOME KDE4 TED2020 XLEnt Ubuntu wikimedia MultiCCAligned; do
  datapath=$basepath/$data
  for sfile in $datapath/*.tsv; do
    value=$data$'\t'$sfile
    if [[ ! " ${donepred[*]} " =~ " ${value} " ]]; then
        lang="$(cut -d'/' -f6 <<<"$sfile")"
        lang=${lang//.tsv}

        model_path=$modelspath/$model/$lang
        for x in $model_path/$model*.pt; do
            loss="$(cut -d'_' -f6 <<<"$x")"
            if (( $(awk <<<"$loss < $val_loss") )); then
                model_path=$model_path/$x
            else
                model_path=""
            fi
        done

        data_to_classify=$sfile
        save_to=$datapath/$model"_"$lang".preds"

        echo "$model_path"
        echo "$data_to_classify"
        echo "$save_to"

        echo "Finished: $data $lang"

    elif [[ " ${donepred[*]} " =~ " ${value} " ]]; then
	    echo "Finished: $data $lang"
    fi
  done
  echo ''
done