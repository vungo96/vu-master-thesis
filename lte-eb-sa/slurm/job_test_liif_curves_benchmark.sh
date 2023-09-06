#!/bin/bash

### Start of Slurm SBATCH definitions
#SBATCH -c 8                # Number of cores (-c)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of t minutes
#SBATCH -p gpu              # Partition to submit to
#SBATCH --mem-per-cpu=5000M #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
#SBATCH --gres=gpu:1
#SBATCH -o slurm/outputs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e slurm/outputs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# name the job
#SBATCH --job-name=test_liif

### end of Slurm SBATCH definitions

tag="rdn-1toMax-div2k-edge-crop-batch-64"
dir=save
benchmark=set5
model_path=_train_rdn-baseline-lte-variable-input-div2k-final_sample-2304-scale-1toMax-inputs-48-div2k-edge-crop-batch-64-resume
iterations=_iterations
max_scale=32
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

python test_all_epochs$iterations.py --config configs/test/test-$benchmark-2.yaml --model_path $dir/$model_path --scale 2 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-3.yaml --model_path $dir/$model_path --scale 3 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-4.yaml --model_path $dir/$model_path --scale 4 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-6.yaml --model_path $dir/$model_path --scale 6 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-8.yaml --model_path $dir/$model_path --scale 8 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-12.yaml --model_path $dir/$model_path --scale 12 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark

benchmark=set14
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-2.yaml --model_path $dir/$model_path --scale 2 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-3.yaml --model_path $dir/$model_path --scale 3 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-4.yaml --model_path $dir/$model_path --scale 4 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-6.yaml --model_path $dir/$model_path --scale 6 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-8.yaml --model_path $dir/$model_path --scale 8 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-12.yaml --model_path $dir/$model_path --scale 12 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark

benchmark=b100
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-2.yaml --model_path $dir/$model_path --scale 2 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-3.yaml --model_path $dir/$model_path --scale 3 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-4.yaml --model_path $dir/$model_path --scale 4 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-6.yaml --model_path $dir/$model_path --scale 6 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-8.yaml --model_path $dir/$model_path --scale 8 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-12.yaml --model_path $dir/$model_path --scale 12 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark

benchmark=urban100
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-2.yaml --model_path $dir/$model_path --scale 2 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-3.yaml --model_path $dir/$model_path --scale 3 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-4.yaml --model_path $dir/$model_path --scale 4 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-6.yaml --model_path $dir/$model_path --scale 6 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-8.yaml --model_path $dir/$model_path --scale 8 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-12.yaml --model_path $dir/$model_path --scale 12 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark

benchmark=manga109
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-2.yaml --model_path $dir/$model_path --scale 2 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-3.yaml --model_path $dir/$model_path --scale 3 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-4.yaml --model_path $dir/$model_path --scale 4 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-6.yaml --model_path $dir/$model_path --scale 6 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-8.yaml --model_path $dir/$model_path --scale 8 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
python test_all_epochs$iterations.py --config configs/test/test-$benchmark-12.yaml --model_path $dir/$model_path --scale 12 --tag $tag --window $window --max_scale $max_scale --dataset $benchmark
