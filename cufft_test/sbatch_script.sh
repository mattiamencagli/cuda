#!/bin/bash
#SBATCH --account cin_staff
#SBATCH --partition boost_usr_prod
#SBATCH --qos boost_qos_dbg
#SBATCH --time 00:20:00            # format: HH:MM:SS
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=32       # n tasks out of 32
#SBATCH --gres=gpu:4               # 1 gpus per node out of 4
#SBATCH --job-name=CUFFT

## To avoid some warnings from the GPU
export OMPI_MCA_btl="^openib"

## DIR
WORK_DIR="${HOME}/programming/cufft_test"
cd $WORK_DIR

## load needed modules on the node
source modules_to_load 

## print GPUs node information
nvidia-smi

## recompile on the node
time make -j -B

## run the test
time ./1d_r2c_example

