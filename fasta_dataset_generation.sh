#!/usr/bin/bash

#SBATCH -J YORO
#SBATCH -D .
#SBATCH -o out_repeat.txt
#SBATCH -e err_repeat.txt
#SBATCH -n 10
#SBATCH -N 1
#SBATCH --partition=long
##SBATCH --gres=gpu:1
##SBATCH --account=coffea_genomes
##SBATCH --time=1-23:00:00
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luish.lopezm@autonoma.edu.co

source ~/.bashrc
conda activate YoloDNA
/shared/home/sorozcoarias/anaconda3/bin/time -f 'Elapsed time: %e s - memory used: %M kb - CPU used: %P' python download_info.py 
