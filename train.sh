#!/bin/sh

# Hard coded settings for resources
# time limit
export ttime=24:00
# number of gpus per job
export num_gpu_per_job=1
# memory per job
export mem_per_gpu=15000


export JOB_NAME='cil'

# load python
module load python_gpu

sh submit_train.sh
