#!/bin/bash

### Start of Slurm SBATCH definitions
#SBATCH -c 8                # Number of cores (-c)
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of t minutes
#SBATCH -p gpu_requeue              # Partition to submit to
#SBATCH --mem-per-cpu=5000M #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
#SBATCH --gres=gpu:1
#SBATCH -o slurm/outputs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e slurm/outputs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# name the job
#SBATCH --job-name=test_liif

### end of Slurm SBATCH definitions

tag="clit-1to4-baseline"
dir=save
model_path=lit_edsr_sample-2304-scale-1to4-batch-32-inputs-48-resume

cd /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/mngo/vu-master-thesis/CLIT

### load modules
module load Anaconda/5.0.1-fasrc02
module load cuda/11.6.2-fasrc01
module load cudnn/8.5.0.96_cuda11-fasrc01
 
### beginning of executable commands
source activate liif_glean_experiment_python3.7_torch1.12.0

python test_all_epochs.py --config configs/test/test-div2k-2.yaml --model_path $dir/$model_path --scale 2 --tag $tag
python test_all_epochs.py --config configs/test/test-div2k-3.yaml --model_path $dir/$model_path --scale 3 --tag $tag
python test_all_epochs.py --config configs/test/test-div2k-4.yaml --model_path $dir/$model_path --scale 4 --tag $tag
python test_all_epochs.py --config configs/test/test-div2k-6.yaml --model_path $dir/$model_path --scale 6 --tag $tag 
python test_all_epochs.py --config configs/test/test-div2k-12.yaml --model_path $dir/$model_path --scale 12 --tag $tag
python test_all_epochs.py --config configs/test/test-div2k-18.yaml --model_path $dir/$model_path --scale 18 --tag $tag  
python test_all_epochs.py --config configs/test/test-div2k-24.yaml --model_path $dir/$model_path --scale 24 --tag $tag 
python test_all_epochs.py --config configs/test/test-div2k-30.yaml --model_path $dir/$model_path --scale 30 --tag $tag 