#!/bin/bash
srun --container-image=/home/vlongle/code/ManiSkill2/lnle+mani_skill2+latest.sqsh\
 --gpus=6\
 --nodes=1\
 --cpus-per-gpu=24\
 --mem-per-cpu=1G\
 --time=72:00:00\
 --qos=ee-med\
 --partition=eaton-compute \
bash -c "python train.py"

exit 3