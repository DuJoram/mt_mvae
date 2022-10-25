#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000
#SBATCH --time=2:00:00
#SBATCH --output=install.out.txt

VENV=~/venvs/mt_mvae_gpu

pip install virtualenv
virtualenv --system-site-package $VENV
source $VENV/bin/activate

pip install -r requirements_gpu.txt

