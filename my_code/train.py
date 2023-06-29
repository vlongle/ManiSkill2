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

trainer_files = ["pn_example.py",
                 "rl_example.py",
                 "rbgd_example.py"]

# trainer_files = ["toy1.py",
#                  "toy2.py",]

# env_ids = ["ENV1", "ENV2", "ENV3", "ENV4"]
seeds = [0, 1, 2, 3]

commands = []
for env_id in env_ids:
    for trainer_file in trainer_files:
        for seed in seeds:
            cmd = f"python {trainer_file} --env_id {env_id} --seed {seed}"
            commands.append(cmd)


max_concurrent_cmds = 8


# Function to run a command
def run_cmd(cmd):
    print(f"Running command: {cmd}")
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()


# Create a pool of workers
with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_cmds) as executor:
    # Map the commands to the workers in the pool
    executor.map(run_cmd, commands)
