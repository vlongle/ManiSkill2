'''
File: /train.py
Project: my_code
Created Date: Thursday June 29th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import platform
from env_ids import env_ids
import concurrent.futures
import subprocess
import torch
import os

# print out the output of nvidia-smi


# # 19 tasks * 4 seeds * 3 obs modes = 228 exps
trainer_files = ["rl_example.py",
                "pn_example.py",
                 "rbgd_example.py"]
# env_ids = ["PickCube-v0"]
# env_ids = ["PickCube-v0", "StackCube-v0"]
# trainer_files = ["toy_example.py"]
# trainer_files = ["toy_example.py"]

# trainer_files = ["toy1.py",
#                  "toy2.py",]
# env_ids = ["ENV1", "ENV2"]
# seeds = [0, 1, 2]
seeds = [0]

commands = []

num_gpus = torch.cuda.device_count()
print("Number of GPUs:", num_gpus)
print("gpu available", torch.cuda.is_available())

for env_id in env_ids:
    for trainer_file in trainer_files:
        for seed in seeds:
            cmd = f"python {trainer_file} --env_id {env_id} --seed {seed}"
            # distribute GPUs to each command
            gpu_id = len(commands) % num_gpus
            # cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python {trainer_file} --env_id {env_id} --seed {seed}"
            cmd = f"python {trainer_file} --env_id {env_id} --seed {seed}"
            commands.append((cmd, gpu_id))  # Also store the GPU id

num_cmds_per_gpu = 2
max_concurrent_cmds = num_cmds_per_gpu * num_gpus


# # Function to run a command
# def run_cmd(cmd):
#     print(f"Running command: {cmd}")
#     process = subprocess.Popen(cmd, shell=True)
#     process.communicate()

# NOTE: https://github.com/haosulab/ManiSkill2/issues/73
# need to unset the DISPLAY

def run_cmd(cmd_and_gpu_id):
    cmd, gpu_id = cmd_and_gpu_id
    my_env = os.environ.copy()
    my_env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"Node {platform.node()}: Running command: {cmd} on GPU {gpu_id}")
    # wait for the print to finish before running the command
    process = subprocess.Popen(cmd, shell=True, env=my_env)
    process.communicate()


# Create a pool of workers
with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_cmds) as executor:
    # Map the commands to the workers in the pool
    executor.map(run_cmd, commands)
