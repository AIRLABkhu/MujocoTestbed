#!/bin/bash

LOGDIR="results"
TASK=$1
ALGO=$2

echo "Experiments started."
for seed in $(seq 0 9)
do
    python mujoco_${ALGO}.py --task $TASK --epoch 200 --seed $seed --logdir $LOGDIR
done
echo "Experiments ended."
