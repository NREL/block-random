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
case "$1" in
# First parameter sweep
1) odir="sweep-1m"
   dataset="shuffled-1m"
   learning_rates=(1e-2 1e-3 1e-4 1e-5)
   layers=(2 4 8 12 16)
   nodes=(8 16 32 64 128 256)
   ;;
# Second parameter sweep
2) odir="sweep"
   dataset="shuffled-16"
   learning_rates=(1e-4)
   layers=(4)
   nodes=(8 16 32 64 128 256)
   ;;
# Third parameter sweep
3) odir="sweep"
   dataset="shuffled-16"
   learning_rates=(1e-4)
   layers=(8 12)
   nodes=(256)
   ;;
# Fourth parameter sweep
4) odir="sweep"
   dataset="shuffled-16"
   learning_rates=(1e-4)
   layers=(2 4)
   nodes=(512)
   ;;    
esac

for learning_rate in "${learning_rates[@]}"
do
    for layer in "${layers[@]}"
    do
        for node in "${nodes[@]}"
        do
            echo "Running lr=$learning_rate, layer=$layer, node=$node"
            pipenv run python learning.py --ldir "$dataset" -o "$odir" --learning_rate "$learning_rate" -l "$layer" -n "$node" &
            sleep 20s
        done
    done
done
wait
