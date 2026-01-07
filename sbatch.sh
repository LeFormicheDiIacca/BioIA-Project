#!/bin/bash
#SBATCH -p edu-medium
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -t 0-01:15
#SBATCH --gres=gpu:0

source /home/${USER}/.bashrc
source activate py313

pip install -r requirements.txt
python test.py
