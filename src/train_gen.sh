#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=MSVA-12
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ab10945@nyu.edu
#SBATCH --output=slurm_logs/slurm_%j.out

module purge

singularity exec --nv --overlay /scratch/ab10945/CV/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/sum_env.sh; conda activate dsnet; python train.py anchor-based --model-dir ../models/ab_basic --splits ../splits/tvsum.yml ../splits/summe.yml"
