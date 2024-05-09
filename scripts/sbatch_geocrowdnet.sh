#!/bin/bash

#SBATCH -J shahana_outlier_geocrowdnet
#SBATCH -A eecs
#SBATCH --time=0-12:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH -p dgx2

#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out

#_SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10

# module load cuda/12.2

RUNPATH=/nfs/stak/users/nguyetr9/hpc-share/shahana_outlier
source ../venvs/pytorch12/bin/activate

echo "python localtion: $(which python)"

# This is the main command
python src/run_geocrowdnet.py -m train.lam=1e-3,1e-4,1e-5

