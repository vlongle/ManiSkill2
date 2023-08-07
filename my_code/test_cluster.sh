#!/bin/bash
srun --container-image=/home/vlongle/code/ManiSkill2/lnle+mani_skill2+latest.sqsh\
 --container-mounts=/home/vlongle/code/ManiSkill2/my_code:/ManiSkill2/my_code\
 --gpus=4\
 --nodes=1\
 --cpus-per-gpu=24\
 --mem-per-cpu=1G\
 --time=72:00:00\
 --qos=ee-med\
 --partition=eaton-compute \
bash -c "python hello.py"

exit 3