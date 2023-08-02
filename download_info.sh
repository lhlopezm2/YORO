#!/usr/bin/bash
#SBATCH -J download
#SBATCH -D .
#SBATCH -o out_download_info.txt
#SBATCH -e err_download_info.txt
#SBATCH -n 10
#SBATCH -N 1
#SBATCH --partition=fast
#SBATCH --mem=15G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luish.lopezm@autonoma.edu.co

source ~/.bashrc
conda activate YoloDNA
/shared/home/sorozcoarias/anaconda3/bin/time -f 'Elapsed time: %e s - memory used: %M kb - CPU used: %P' python download_info.py 
