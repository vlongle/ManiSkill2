#!/bin/bash
# run train_cluster.sh and saves a copy of the screen output to `log.txt` file
bash train_cluster.sh / 2>&1 | tee log.txt
