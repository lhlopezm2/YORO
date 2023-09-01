#!/usr/bin/bash
#SBATCH -J YORO
#SBATCH -D .
#SBATCH -o out_training.txt
#SBATCH -e err_training.txt
#SBATCH -n 15
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luish.lopezm@autonoma.edu.co

module load tensorflow-gpu/2.6.2
/shared/home/sorozcoarias/anaconda3/bin/time -f 'Elapsed time: %e s - memory used: %M kb - CPU used: %P' python training.py 
