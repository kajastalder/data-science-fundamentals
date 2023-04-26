#!/bin/bash
#SBATCH --job-name="Midterm Project"
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=2GB

module purge
module load Python

pip install --user rdkit scikit-learn tensorflow numpy pandas matplotlib
srun python3 ./DSF/midterm.py
