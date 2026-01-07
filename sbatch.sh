#!/bin/bash
#SBATCH -p short
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -t 0-00:15

source /home/${USER}/.bashrc
source activate py313

pip install -r requirements.txt
python test.py
