#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --job-name=nanoGPT
#SBATCH --output=my_job_%j.out

# Change directory
cd /u/dssc/valinsogna/nanoGPT

# Commands to be executed
torchrun --standalone --nproc_per_node=4 train.py config/train_whatsapp.py

#As soon as it finishes:
exit