#!/bin/bash

### Start of Slurm SBATCH definitions
#SBATCH -c 14               # Number of cores (-c)
#SBATCH -t 2-00:00  # 2-08:00          # Runtime in D-HH:MM, minimum of t minutes
#SBATCH -p gpu_requeue              # Partition to submit to
#SBATCH --mem-per-cpu=20G #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
#SBATCH --gres=gpu:4
#SBATCH -o slurm/outputs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e slurm/outputs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# name the job
#SBATCH --job-name=train_liif_glean

### end of Slurm SBATCH definitions

cd /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/mngo/vu-master-thesis/CLIT

### load modules
module load Anaconda/5.0.1-fasrc02
module load cuda/11.6.2-fasrc01
module load cudnn/8.5.0.96_cuda11-fasrc01
 
### beginning of executable commands
source activate liif_glean_experiment_python3.7_torch1.12.0
python train.py --config configs/train/train_edsr_baseline_clit2.yaml --name clit2_edsr --tag sample-2304-scales-1to4-1to4-batch-32-inputs-48