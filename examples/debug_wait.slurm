#!/bin/bash
#SBATCH --job-name=DebugWait
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=3920G
#SBATCH --cpus-per-task=128
#SBATCH -A casus
#SBATCH -p casus

source ~/jupyter.profile

python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client $@


