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
python eval.py \
 --checkpoint=/home/mila/c/chris.emezue/scratch/wmt22/unfiltered/wmt22_mmtafrica_unfiltered.pt \
 --parallel_dir=/home/mila/c/chris.emezue/scratch/wmt22/data/mmt-africa-format/unfiltered \
 --homepath=/home/mila/c/chris.emezue/scratch/wmt22/unfiltered \
 