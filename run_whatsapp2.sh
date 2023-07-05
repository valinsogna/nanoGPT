#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --job-name=nanoGPT2
#SBATCH --output=my_job_%j.out

# Change directory
cd /u/dssc/valinsogna/nanoGPT

pip install torch numpy transformers datasets tiktoken wandb tqdm

# Commands to be executed
python3 train.py config/train_whatsapp2.py

#As soon as it finishes:
exit