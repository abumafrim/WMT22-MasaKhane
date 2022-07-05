#!/bin/bash
#SBATCH --job-name=wmt22_mmtafrica_download
#SBATCH --mem=50GB               # memory (per node)
#SBATCH --time=1-5:50            # time (DD-HH:MM)
#SBATCH --error=/home/mila/c/chris.emezue/wmt22/downloaderror-%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/wmt22/downloadoutput-%j.txt

###########cluster information above this line


cd ~/scratch/wmt22/data

gdrive download 18CyH7DmdaedgGXZJdIrA3MaY0LKZUo9q --recursive 

echo 'ALL DONE'