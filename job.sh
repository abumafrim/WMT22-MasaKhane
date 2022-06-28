#!/bin/bash
#SBATCH --job-name=wmt22_mmtafrica
#SBATCH --gres=gpu:48gb:1             # Number of GPUs (per node)
#SBATCH --mem=100G               # memory (per node)
#SBATCH --time=4-5:50            # time (DD-HH:MM)
#SBATCH --error=/home/mila/c/chris.emezue/wmt22/slurmerror-%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/wmt22/slurmoutput-%j.txt

###########cluster information above this line


###load environment 


module load python/3
source ~/scratch/wmt22/wmtenv/bin/activate   

CURR_DIR=/home/mila/c/chris.emezue/wmt22
cd mmtafrica
python mmtafrica.py \
 --parallel_dir=data/parallel \
 --homepath=/home/mila/c/chris.emezue/scratch/wmt22 \
 --print_freq=100 \
 --use_reconstruction=False \
 --do_backtranslation=False \
 --checkpoint_freq=15_000 \
 --model_name=wmt22_mmtafrica_final \
 --n_epochs=20 \
 --gradient_accumulation_batch=256 \
 --batch_size=16 \
 --log=${CURR_DIR}/train.log \
 --lr=1e-4