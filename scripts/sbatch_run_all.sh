#!/bin/bash

#SBATCH -J stl10
#SBATCH -A eecs
#SBATCH --time=0-12:00:00
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
python src/run_reweight.py data=stl10_machine_3 with_ray=False; \
python src/run_reweight.py data=stl10_machine_3 with_ray=False; \
python src/run_reweight.py data=stl10_machine_3 with_ray=False; \
python src/run_volminnet.py data=stl10_machine_3 with_ray=False; \
python src/run_bltm.py data=stl10_machine_3 with_ray=False; \
python src/run_ptd.py data=stl10_machine_3 with_ray=False; \
python src/run_meidtm.py data=stl10_machine_3 with_ray=False; \
python src/run_geocrowdnet.py data=stl10_machine_3 with_ray=False train=geocrowdnetf; \
python src/run_geocrowdnet.py data=stl10_machine_3 with_ray=False train=geocrowdnetw; \
python src/run_maxmig.py data=stl10_machine_3 with_ray=False; \
python src/run_tracereg.py data=stl10_machine_3 with_ray=False; \
python src/run_crowdlayer.py data=stl10_machine_3 with_ray=False; \
python src/my_training.py data=stl10_machine_3 with_ray=False; \
echo "done"

