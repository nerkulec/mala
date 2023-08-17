#!/bin/bash
#SBATCH --job-name=Train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=370G
#SBATCH --cpus-per-task=48
#SBATCH -A casus
#SBATCH -p casus

source ~/jupyter.profile

python3 $@


