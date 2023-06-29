'''
File: /train.py
Project: my_code
Created Date: Thursday June 29th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
from env_ids import env_ids
import concurrent.futures
import subprocess
import torch
import os

trainer_files = ["pn_example.py",]


seeds = [0, 1]
env_ids = ["LiftCube-v0"]


commands = []

num_gpus = torch.cuda.device_count()
print("num_gpus", num_gpus)

for env_id in env_ids:
    for trainer_file in trainer_files:
        for seed in seeds:
            cmd = f"python {trainer_file} --env_id {env_id} --seed {seed}"
            # distribute GPUs to each command
            gpu_id = len(commands) % num_gpus
            # cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python {trainer_file} --env_id {env_id} --seed {seed}"
            cmd = f"python {trainer_file} --env_id {env_id} --seed {seed}"
            commands.append((cmd, gpu_id))  # Also store the GPU id

# from pprint import pprint
# print("commands", len(commands))
# pprint(commands)
max_concurrent_cmds = 8


# # Function to run a command
# def run_cmd(cmd):
#     print(f"Running command: {cmd}")
#     process = subprocess.Popen(cmd, shell=True)
#     process.communicate()

def run_cmd(cmd_and_gpu_id):
    cmd, gpu_id = cmd_and_gpu_id
    my_env = os.environ.copy()
    my_env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"Running command: {cmd} on GPU {gpu_id}")
    process = subprocess.Popen(cmd, shell=True, env=my_env)
    process.communicate()


# Create a pool of workers
with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_cmds) as executor:
    # Map the commands to the workers in the pool
    executor.map(run_cmd, commands)
