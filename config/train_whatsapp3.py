# Train on 1 GPU

#####################################################################
# CHANGED default parameters from train.py model:
#####################################################################
out_dir = 'out-whatsapp3'
dataset = 'whatsapp'
wandb_project = 'whatsapp3'
wandb_run_name='gpt2-124M-3'

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# bias = False
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually

# eval stuff
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often
# this makes total number of tokens be 300B (recalculate for whatsapp)
warmup_iters = 100

gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

#####################################################################
# UNCHANGED default parameters from train.py model:
#####################################################################
wandb_log = False

# weight decay
weight_decay = 1e-1
