NEW_HF_DIR=/home/mila/c/chris.emezue/scratch/wmt22/data/hugging_face_mmtafrica

CURR_DIR=/home/mila/c/chris.emezue/scratch/wmt22/data/huggingface_laser
cd $CURR_DIR



for file in *
do
    #Read the split words into an array based on comma delimiter
    IFS='-'

    read -a strarr <<< $file
    first_part=${strarr[0]}
    tgt=${strarr[1]}
    IFS='_'
    read -a strarr2 <<< $first_part
    IFS='|'
    new_file_name="${strarr2[2]}_${tgt}_wmt22_african_para.tsv"
    #echo "$new_file_name | ${file}"
    echo $file
    cp $file ${NEW_HF_DIR}/${new_file_name}   

done 
echo "All Done"