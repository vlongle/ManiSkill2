#!/bin/bash
srun --partition=eaton-compute\
  --qos=ee-med\
  --gpus=1\
  --mem-per-gpu=16GB\
 --cpus-per-gpu=8 \
 enroot import docker://lnle/mani_skill2:latest
