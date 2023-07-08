#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --job-name=nanoGPT3
#SBATCH --output=my_job_%j.out

# Change directory
cd /u/dssc/valinsogna/nanoGPT

pip install torch numpy transformers datasets tiktoken wandb tqdm

# Commands to be executed
torchrun --standalone --nproc_per_node=1 train.py config/train_whatsapp3.py

#As soon as it finishes:
exit