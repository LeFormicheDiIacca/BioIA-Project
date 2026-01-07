#!/bin/bash
#SBATCH -p short
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -t 0-00:15
#SBATCH --gres=gpu:0

source /home/${USER}/.bashrc
source activate py313

pip install -r requirements.txt
python test.py
