#!/bin/bash

### Start of Slurm SBATCH definitions
#SBATCH -c 8                # Number of cores (-c)
#SBATCH -t 0-01:00          # Runtime in D-HH:MM, minimum of t minutes
#SBATCH -p gpu              # Partition to submit to
#SBATCH --mem-per-cpu=5000M #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
#SBATCH --gres=gpu:1
#SBATCH -o slurm/outputs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e slurm/outputs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# name the job
#SBATCH --job-name=test_liif

### end of Slurm SBATCH definitions

tag="chosen-dataset"
dir=save
model_path=_train_edsr-baseline-lte-variable-input-flickr2k-gan8_sample-2304-scale-1toMax-inputs-48-lsdir-gan-based-huber-edge-crop-edge-sample-finetune-1toMax-flickr2k-edge-crop-lpips-e3-batch-128-lr-e4-resume
iterations=_iterations
max_scale=4
window=0

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

python test_all_epochs$iterations.py --config configs/test/test-chosen-18.yaml --model_path $dir/$model_path --scale 18 --tag $tag --window $window --max_scale $max_scale --out_dir test_images/
#python test_all_epochs$iterations.py --config configs/test/test-chosen-24.yaml --model_path $dir/$model_path --scale 24 --tag $tag --window $window --max_scale $max_scale --out_dir test_images/
#python test_all_epochs$iterations.py --config configs/test/test-chosen-30.yaml --model_path $dir/$model_path --scale 30 --tag $tag --window $window --max_scale $max_scale --out_dir test_images/
