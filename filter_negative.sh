#!/usr/bin/bash

#SBATCH -J filter
#SBATCH -D .
#SBATCH -o out_filter_negative.txt
#SBATCH -e err_filter_negative.txt
#SBATCH -n 5
#SBATCH -N 1
#SBATCH --partition=fast
#SBATCH --mem=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luish.lopezm@autonoma.edu.co

source ~/.bashrc
conda activate bioinfo
/shared/home/sorozcoarias/anaconda3/bin/time -f 'Elapsed time: %e s - memory used: %M kb - CPU used: %P' python filter_negative.py 
