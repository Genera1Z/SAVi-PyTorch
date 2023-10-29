#!/bin/bash
#SBATCH --job-name=savi
#SBATCH --account=project_2008396
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpusmall
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --gres=gpu:a100:1,nvme:25

module load pytorch tensorflow vim
pip install -r requirements.txt
cp -r /scratch/project_2008396/Datasets/movi_a/* $LOCAL_SCRATCH

srun python main.py --data_dir $LOCAL_SCRATCH
srun python main.py --data_dir $LOCAL_SCRATCH
srun python main.py --data_dir $LOCAL_SCRATCH
srun python main.py --data_dir $LOCAL_SCRATCH
srun python main.py --data_dir $LOCAL_SCRATCH
