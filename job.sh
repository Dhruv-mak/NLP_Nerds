#!/bin/bash
#SBATCH --job-name=vasptest
#SBATCH --output=vasp.out
#SBATCH --error=vasp.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dmakwana@ufl.edu
#SBATCH --partition=gpu
#SBATCH --gpus=a100:3
#SBATCH --time=15000

module purge
module load cuda/12.1.1  intel openmpi
mamba init bash
mamba activate nlp
python some.py
