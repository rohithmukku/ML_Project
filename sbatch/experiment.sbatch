#!/bin/bash

#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=40GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=experiment_%j.out
#SBATCH --error=experiment_%j.err

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER


singularity exec --nv \
--overlay $SCRATCH/full_time/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "

source /ext3/env.sh
conda activate full_time
cd $SCRATCH/ml/ML_Project/test_files/
python experiment.py --batch_size=128 --indata_size=45000 --outdata_size=6000 --test_indata_size=900 --test_outdata_size=300
"
