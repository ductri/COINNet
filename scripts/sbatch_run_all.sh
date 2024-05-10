#!/bin/bash

#SBATCH -J run_all
#SBATCH -A eecs
#SBATCH --time=0-10:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH -p dgxh,dgx2,gpu

#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10

module load cuda/12.2

RUNPATH=/nfs/stak/users/nguyetr9/hpc-share/shahana_outlier
source ../venvs/pytorch12/bin/activate

echo "python location: $(which python)"

# This is the main command
python src/run_crowdlayer.py data=cifar10_machine_6 with_ray=False; \
python src/run_crowdlayer.py data=cifar10_machine_7 with_ray=False; \
python src/run_tracereg.py data=cifar10_machine_6 with_ray=False; \
python src/run_tracereg.py data=cifar10_machine_7 with_ray=False; \
python src/run_geocrowdnet.py  train=geocrowdnetw data=cifar10_machine_6; \
python src/run_geocrowdnet.py  train=geocrowdnetw data=cifar10_machine_7; \
echo Done

