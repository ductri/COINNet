#!/bin/bash

#SBATCH -J our
#SBATCH -A eecs
#SBATCH --time=0-04:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH -p dgx2,gpu

#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15

module load cuda/12.2
source ../venvs/pytorch12/bin/activate

# This is the main command
python src/my_training.py with_ray=False data=fmnist_machine_2;
python src/my_training.py with_ray=False data=fmnist_machine_3;
python src/my_training.py with_ray=False data=fmnist_machine_2;
python src/my_training.py with_ray=False data=fmnist_machine_3;
echo "done"

