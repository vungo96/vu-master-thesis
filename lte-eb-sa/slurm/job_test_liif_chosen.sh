#!/bin/bash

### Start of Slurm SBATCH definitions
#SBATCH -c 8                # Number of cores (-c)
#SBATCH -t 0-08:00          # Runtime in D-HH:MM, minimum of t minutes
#SBATCH -p gpu              # Partition to submit 
#SBATCH --mem-per-cpu=5000M #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
#SBATCH --gres=gpu:1
#SBATCH -o slurm/outputs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e slurm/outputs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# name the job
#SBATCH --job-name=test_liif

### end of Slurm SBATCH definitions

# model=swinir-traditional/iteration-1000000.pth
model=_train_swinir-baseline-lte-variable-input-lsdir-final-new2_sample-2304-scale-1to4-inputs-48-lsdir-edge-crop-batch-64-resume/iteration-996000.pth
#model=_train_swinir-baseline-lte-variable-input-lsdir-final-new-traditional_sample-2304-scale-1to4-inputs-48-lsdir-traditional-resume/iteration-982720.pth
tag=lsdir-1to4-edge-crop
dir=save
max_scale=4 # training scale
window=8

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

python test.py --config configs/test/test-chosen-2.yaml --model $dir/$model --gpu 0 --tag $tag --max_scale $max_scale --window $window --out_dir test_images-x2/
python test.py --config configs/test/test-chosen-3.yaml --model $dir/$model --gpu 0 --tag $tag --max_scale $max_scale --window $window --out_dir test_images-x3/
python test.py --config configs/test/test-chosen-4.yaml --model $dir/$model --gpu 0 --tag $tag --max_scale $max_scale --window $window --out_dir test_images-x4/
python test.py --config configs/test/test-chosen-6.yaml --model $dir/$model --gpu 0 --tag $tag --max_scale $max_scale --window $window --out_dir test_images-x6/
python test.py --config configs/test/test-chosen-8.yaml --model $dir/$model --gpu 0 --tag $tag --max_scale $max_scale --window $window --out_dir test_images-x8/
python test.py --config configs/test/test-chosen-12.yaml --model $dir/$model --gpu 0 --tag $tag --max_scale $max_scale --window $window --out_dir test_images-x12/
python test.py --config configs/test/test-chosen-18.yaml --model $dir/$model --gpu 0 --tag $tag --max_scale $max_scale --window $window --out_dir test_images-x18/
python test.py --config configs/test/test-chosen-24.yaml --model $dir/$model --gpu 0 --tag $tag --max_scale $max_scale --window $window --out_dir test_images-x24/
python test.py --config configs/test/test-chosen-30.yaml --model $dir/$model --gpu 0 --tag $tag --max_scale $max_scale --window $window --out_dir test_images-x30/

#benchmark=urban100
#python test.py --config configs/test/test-$benchmark-4.yaml --model $dir/$model --gpu 0 --tag $tag --window $window --max_scale $max_scale --out_dir test_images/


#python test.py --config configs/test/test-urban100-6.yaml --model $dir/$model --gpu 0 --tag $tag-last --max_scale $max_scale --window $window --out_dir test_images/

#python test.py --config configs/test/test-urban100-12.yaml --model $dir/$model --gpu 0 --tag $tag-last --max_scale $max_scale --window $window --out_dir test_images/
