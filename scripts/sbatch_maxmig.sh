#!/bin/bash

#SBATCH -J shahana_outlier_maxmig
#SBATCH -A eecs
#SBATCH --time=0-16:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH -p dgxh,dgxs,gpu

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
python src/run_maxmig.py train=maxmig data=cifar10_syn \
    data.total_noise_rate=0.2 data.percent_instance_noise=0.1 &&
python src/run_maxmig.py train=maxmig data=cifar10_syn \
    data.total_noise_rate=0.2 data.percent_instance_noise=0.3 &&
python src/run_maxmig.py train=maxmig data=cifar10_syn \
    data.total_noise_rate=0.4 data.percent_instance_noise=0.1 &&
python src/run_maxmig.py train=maxmig data=cifar10_syn \
    data.total_noise_rate=0.4 data.percent_instance_noise=0.3 &&
python src/run_maxmig.py train=maxmig data=cifar10_syn \
    data.total_noise_rate=0.4 data.percent_instance_noise=0.5 

