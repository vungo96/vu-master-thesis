#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=$((12000 + $RANDOM % 20000))

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    "$(dirname "$0")"/test.py $CONFIG $CHECKPOINT --launcher pytorch "${@:4}"
