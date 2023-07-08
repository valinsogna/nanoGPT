#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --job-name=finetuneWhatsApp
#SBATCH --output=my_job_%j.out

# Change directory
cd /u/dssc/valinsogna/nanoGPT

pip install torch numpy transformers datasets tiktoken wandb tqdm

# Commands to be executed
torchrun --standalone --nproc_per_node=4 train.py config/finetune_whatsapp.py

#As soon as it finishes:
exit