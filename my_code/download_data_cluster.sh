#!/bin/bash
srun --container-image=/home/vlongle/code/ManiSkill2/lnle+mani_skill2+latest.sqsh\
 --container-mounts=/home/vlongle/code/ManiSkill2/my_code:/ManiSkill2/my_code\
 --gpus=1\
 --nodes=1\
 --cpus-per-gpu=24\
 --mem-per-cpu=1G\
 --time=72:00:00\
 --qos=ee-med\
 --partition=eaton-compute \
bash -c "python -m mani_skill2.utils.download_demo all"

exit 3