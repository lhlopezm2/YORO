#!/usr/bin/bash
#SBATCH -J YORO
#SBATCH -D .
#SBATCH -o out_npy_generation.txt
#SBATCH -e err_npy_generation.txt
#SBATCH -n 51
#SBATCH -N 1
#SBATCH --partition=fast
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luish.lopezm@autonoma.edu.co

source ~/.bashrc
conda activate YORO
/shared/home/sorozcoarias/anaconda3/bin/time -f 'Elapsed time: %e s - memory used: %M kb - CPU used: %P' python npy_generation.py 
