#!/bin/bash

#SBATCH --job-name=learning
#SBATCH --account=exact
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH -o %x.o%j

# load modules
module purge
module unuse /nopt/nrel/apps/modules/default/modulefiles
module unuse /usr/share/Modules/modulefiles
module unuse /nopt/modulefiles
module use /nopt/nrel/ecom/hpacf/compilers/modules
module use /nopt/nrel/ecom/hpacf/utilities/modules
module use /nopt/nrel/ecom/hpacf/software/modules/gcc-7.3.0
module load gcc
module load python/3.6.5
module load py-setuptools/40.4.3-py3
module load cuda
module load cudnn

# Run the code
learning_rate=1e-4
layer=4
node=128
datasets=("shuffled-16" "sorted-16" "block-16")
batch_sizes=(128 256 512 1024 2048 4096)
rtypes=("" "uniform" "kmeans")
odir="model_runs"
for dataset in "${datasets[@]}"
do
    for batch in "${batch_sizes[@]}"
    do
        for rtype in "${rtypes[@]}"
        do
            echo "Running lr=$learning_rate, layer=$layer, node=$node, batch_size=$batch, dataset=$dataset, rtype=$rtype"
            pipenv run python learning.py --ldir "$dataset" -o "$odir" --learning_rate "$learning_rate" -l "$layer" -n "$node" -b "$batch" --rebalance "$rtype" &
            sleep 20s
        done
    done
done
wait
