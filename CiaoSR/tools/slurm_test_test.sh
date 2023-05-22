#!/bin/bash

### Start of Slurm SBATCH definitions
#SBATCH -c 14               # Number of cores (-c)
#SBATCH -t 2-00:00          # Runtime in D-HH:MM, minimum of t minutes
#SBATCH -p gpu_requeue              # Partition to submit to
#SBATCH --mem-per-cpu=20000M #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
# --nodes=1              # number of nodes
#SBATCH --ntasks-per-node=2     # MPI processes per node
#SBATCH --gres=gpu:4
#SBATCH --ntasks=2
#SBATCH -o tools/outputs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e tools/outputs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# name the job
#SBATCH --job-name=train_ciaosr

### end of Slurm SBATCH definitions

cd /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/mngo/vu-master-thesis/CiaoSR

### load modules
module load Anaconda/5.0.1-fasrc02
module load gcc/8.3.0-fasrc01
module load cuda/11.1.0-fasrc01
module load cudnn
 
### beginning of executable commands
source activate ciaosr6

export MASTER_PORT=$((12000 + $RANDOM % 20000))

CONFIG=configs/ciaosr/001_localimplicitsr_edsr_div2k_g1_c64b16_1000k_unfold_lec_mulwkv_res_nonlocal.py
GPUS=4
CKPT=work_dirs/001_localimplicitsr_edsr_div2k_g1_c64b16_1000k_unfold_lec_mulwkv_res_nonlocal/latest.pth

PYTHONPATH=/bin/..:tools/..:
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$MASTER_PORT \
    ./tools/test.py $CONFIG $CKPT --launcher pytorch
