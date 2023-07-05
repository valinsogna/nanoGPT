# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

#####################################################################
# CHANGED default parameters from train.py model:
#####################################################################
out_dir = 'out-whatsapp'
dataset = 'whatsapp'
wandb_log = True
wandb_project = 'whatsapp'
wandb_run_name='gpt2-124M'

# bias = False
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# eval stuff
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

#####################################################################
# UNCHANGED default parameters from train.py model:
#####################################################################
# these make the total batch size be ~0.5M (recalculate for whatsapp)
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520 (recalculate for whatsapp)
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B (recalculate for whatsapp)
max_iters = 600000
lr_decay_iters = 600000

# weight decay
weight_decay = 1e-1
