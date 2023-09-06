#!/bin/bash

### Start of Slurm SBATCH definitions
#SBATCH -c 14               # Number of cores (-c)
#SBATCH -t 0-08:00          # Runtime in D-HH:MM, minimum of t minutes
#SBATCH -p seas_gpu         # Partition to submit to
#SBATCH --mem-per-cpu=35G #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
#SBATCH --gres=gpu:1
#SBATCH -o slurm/outputs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e slurm/outputs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# name the job
#SBATCH --job-name=train_liif

### end of Slurm SBATCH definitions

cd /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/mngo/vu-master-thesis/liif

### load modules
# module load Anaconda/5.0.1-fasrc02
# module load Anaconda2/2019.10-fasrc01
#module load cuda/11.6.2-fasrc01
#module load cudnn/8.5.0.96_cuda11-fasrc01
module load cuda/12.0.1-fasrc01
module load cudnn/8.8.0.121_cuda12-fasrc01
 
### beginning of executable commands
source activate liif_glean_experiment_python3.7_torch1.12.0

python train_iter.py --config configs/train-div2k/train_edsr-baseline-lte-variable-input-div2k-final2-test-edge-map.yaml --gpu 0 --tag test-edge-maps