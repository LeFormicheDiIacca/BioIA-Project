#!/bin/sh
#SBATCH -p short
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1024M
#SBATCH -N 1
#SBATCH -t 0-00:10

module load cuda

./simpleDMA 200000000
