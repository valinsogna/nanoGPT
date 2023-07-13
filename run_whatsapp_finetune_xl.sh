#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --job-name=finetuneWhatsApp
#SBATCH --output=my_job_%j.out
#SBATCH --mem=100gb

# Change directory
cd /u/dssc/valinsogna/nanoGPT

pip install torch numpy transformers datasets tiktoken wandb tqdm
#update packages 
pip install --upgrade urllib3
pip install --upgrade chardet

# Commands to be executed
torchrun --standalone --nproc_per_node=8 train.py config/finetune_whatsapp_xl.py

#As soon as it finishes:
exit