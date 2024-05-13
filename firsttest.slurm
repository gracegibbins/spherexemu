#!/bin/bash

#SBATCH --job-name=firsttest
#SBATCH --output=firsttest.out
#SBATCH --account=timeifler
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5gb
#SBATCH --time=00:05:00

#module load python/3.11 anaconda

#module load openmpi3/3.1.4

source /home/u14/gibbins/.bashrc

#conda activate spherexemu

python linps.py
