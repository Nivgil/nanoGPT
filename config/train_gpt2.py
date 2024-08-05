# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

import datetime

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
# 8 batch size * 1024 block size * 8 gradaccum * 8 GPUs = 524,288
# 16 batch size * 1024 block size * 8 gradaccum * 8 GPUs = 2 * 524,288
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 16 * 8

# this (600_000) makes total number of tokens be 300B
# this (200_000) makes total number of tokens be 100B
# this (100_000) makes total number of tokens be 100B
max_iters = 100_000
lr_decay_iters = 100_000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# simulated all-gather drops
drop_prob = 0.0
sampling_method = 'structured_uniform'

# simulated number of workers
sim_world_size = 128

# output directory
timestamp = datetime.datetime.now().strftime("%d-%m-%y_%H-%M")
drop_str = str(drop_prob).replace(".", "_")
out_dir = f'/software/users/ngiladi/nanoGPT/runs/n-{sim_world_size}_{timestamp}_drop-{drop_str}'

wandb_log = True
wandb_project = 'owt'
wandb_run_name = f'gpt2-124M-{sim_world_size}-{int(drop_prob * 100)}'
