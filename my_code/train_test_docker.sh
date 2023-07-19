#!/bin/bash
srun --container-image=/home/vlongle/code/ManiSkill2/lnle+mani_skill2+latest.sqsh\
 --gpus=4\
 --nodes=1\
 --cpus-per-gpu=24\
 --mem-per-cpu=1G\
 --time=72:00:00\
 --qos=ee-med\
 --partition=eaton-compute \
bash -c "python rl_example.py --seed 0 --env_id LiftCube-v0 --total_timesteps 4000 --log_dir test_onev1_logs"

exit 3
