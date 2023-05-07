#!/usr/bin/env bash

set -x

PARTITION=gpu_requeue
JOB_NAME=test-ciaosr
CONFIG=configs/ciaosr/001_localimplicitsr_edsr_div2k_g1_c64b16_1000k_unfold_lec_mulwkv_res_nonlocal.py
CHECKPOINT=work_dirs/001_localimplicitsr_edsr_div2k_g1_c64b16_1000k_unfold_lec_mulwkv_res_nonlocal/latest.pth
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
PY_ARGS=${@:5}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    -t 1-00:00 \
    -c 10 \
    --job-name=${JOB_NAME} \
    --ntasks=${GPUS} \
    --mem-per-cpu=20000M \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" ${PY_ARGS}
