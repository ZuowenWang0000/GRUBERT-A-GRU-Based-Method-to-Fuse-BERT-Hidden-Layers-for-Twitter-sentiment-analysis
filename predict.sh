#!/bin/sh

# Hard coded settings for resources
# time limit
export ttime=4:00
# number of gpus per job
export num_gpu_per_job=1
# memory per job
export mem_per_gpu=22000
# type of GPU
export gpu_type=TeslaV100_SXM2_32GB

export JOB_NAME='cil'

# load python
module load eth_proxy
module load gcc/6.3.0 python_gpu/3.7.4 hdf5

sh submit_predict.sh $@
