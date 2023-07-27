#!/usr/bin/bash

#SBATCH -J YORO
#SBATCH -D .
#SBATCH -o output.txt
#SBATCH -e error.txt
#SBATCH -n 5
#SBATCH -N 1
#SBATCH --partition=long
#SBATCH --mem=3G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luish.lopezm@autonoma.edu.co

source ~/.bashrc
conda activate YoloDNA
/shared/home/sorozcoarias/anaconda3/bin/time -f 'Elapsed time: %e s - memory used: %M kb - CPU used: %P' python npy_dataset_generation.py 
