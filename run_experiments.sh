#!/bin/bash

LOGDIR="results"
TASK=$1
ALGO=$2

echo "Experiments started."
for seed in $(seq 0 9)
do
    python mujoco_${ALGO}.py --step-per-epoch 5000 --task $TASK --epoch 200 --seed $seed --logdir $LOGDIR
done
echo "Experiments ended."

python ./tools --root-dir ./results/$TASK/$ALGO
