#!/bin/bash

### Start of Slurm SBATCH definitions
#SBATCH -c 8                # Number of cores (-c)
#SBATCH -t 0-08:00          # Runtime in D-HH:MM, minimum of t minutes
#SBATCH -p gpu_test              # Partition to submit 
#SBATCH --mem-per-cpu=5000M #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
#SBATCH --gres=gpu:1
#SBATCH -o slurm/outputs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e slurm/outputs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# name the job
#SBATCH --job-name=test_liif

### end of Slurm SBATCH definitions

tag=lit_edsr_sample-2304-scale-1to4-batch-32-inputs-48-resume
dir=save
scale=""

cd /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/mngo/vu-master-thesis/CLIT

### load modules
module load Anaconda/5.0.1-fasrc02
module load cuda/11.6.2-fasrc01
module load cudnn/8.5.0.96_cuda11-fasrc01
 
### beginning of executable commands
source activate liif_glean_experiment_python3.7_torch1.12.0

#python test.py --config configs/test/test-div2k-2$scale.yaml --model $dir/$tag/epoch-best.pth --name $tag
#python test.py --config configs/test/test-div2k-3$scale.yaml --model $dir/$tag/epoch-best.pth --name $tag
#python test.py --config configs/test/test-div2k-4$scale.yaml --model $dir/$tag/epoch-best.pth --name $tag
#python test.py --config configs/test/test-div2k-6$scale.yaml --model $dir/$tag/epoch-best.pth --name $tag
#python test.py --config configs/test/test-div2k-12$scale.yaml --model $dir/$tag/epoch-best.pth --name $tag
#python test.py --config configs/test/test-div2k-18$scale.yaml --model $dir/$tag/epoch-best.pth --name $tag
#python test.py --config configs/test/test-div2k-24$scale.yaml --model $dir/$tag/epoch-best.pth --name $tag
#python test.py --config configs/test/test-div2k-30$scale.yaml --model $dir/$tag/epoch-best.pth --name $tag

#python test.py --config configs/test/test-div2k-2$scale.yaml --model $dir/$tag/epoch-last.pth --name $tag-last
#python test.py --config configs/test/test-div2k-3$scale.yaml --model $dir/$tag/epoch-last.pth --name $tag-last
#python test.py --config configs/test/test-div2k-4$scale.yaml --model $dir/$tag/epoch-last.pth --name $tag-last
python test.py --config configs/test/test-div2k-6$scale.yaml --model $dir/$tag/epoch-last.pth --name $tag-last
python test.py --config configs/test/test-div2k-12$scale.yaml --model $dir/$tag/epoch-last.pth --name $tag-last
python test.py --config configs/test/test-div2k-18$scale.yaml --model $dir/$tag/epoch-last.pth  --name $tag-last
python test.py --config configs/test/test-div2k-24$scale.yaml --model $dir/$tag/epoch-last.pth  --name $tag-last
python test.py --config configs/test/test-div2k-30$scale.yaml --model $dir/$tag/epoch-last.pth  --name $tag-last