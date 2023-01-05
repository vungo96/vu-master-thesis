#!/bin/bash

### Start of Slurm SBATCH definitions
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 0-08:00          # Runtime in D-HH:MM, minimum of t minutes
#SBATCH -p gpu_test              # Partition to submit to
#SBATCH --mem-per-cpu=3900M #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
#SBATCH --gres=gpu:1
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# name the job
#SBATCH --job-name=train_liif_test

### end of Slurm SBATCH definitions

cd /n/home08/mngo/vu-master-thesis/liif

### load modules
module load Anaconda3/2020.11
module load cuda/11.6.2-fasrc01
module load cudnn/8.5.0.96_cuda11-fasrc01
 
### beginning of executable commands
source activate liif_glean_experiment_python3.7_torch1.12.0

python train_liif.py --config configs/train-celebAHQ/train_celebAHQ-32-256_liif.yaml --gpu 1