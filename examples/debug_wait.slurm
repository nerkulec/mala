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

source ~/jupyter.profile

fireSignal()
{
	for i in "$@"
	do
		kill -s $i $PYTHON_PID
		echo "batch script: send signal $1 to $PYTHON_PID" >&2
	done
}

trap "fireSignal SIGUSR1" SIGTERM
trap "fireSignal SIGUSR1" SIGCONT
trap "fireSignal SIGUSR1" SIGUSR1
trap "fireSignal SIGUSR2" SIGINT


python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client $@ &
PYTHON_PID=$!
echo "python PID is $PYTHON_PID"
wait

