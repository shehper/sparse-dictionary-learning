#!/bin/bash
#SBATCH --partition=gpu          # Partition (job queue)
##SBATCH --exclusive
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=nanogpt-owt       # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --cpus-per-task=4        # Cores per task (>1 if multithread tasks)
#SBATCH --mem=4000                # Real memory (RAM) required (MB)
#SBATCH --time=03:00:00           # Total run time limit (HH:MM:SS)
#SBATCH --output=./slurm/slurm.%j.out
#SBATCH --error=./slurm/slurm.%j.err
#SBATCH --gres=gpu:1

#SBATCH --exclude=cuda[001-008],gpu[005-008],pascal[001-010],volta[001-003]  # Sung Hak's rec: 2080 Ti, 3090 and A100 GPUs. 
##SBATCH --exclude=cuda[001-008],gpu[005-014],gpu[017-018],pascal[001-010],volta[001-003] # A100s only
##SBATCH --exclude=cuda[001-008],gpu[005-006],gpu[015-026],pascal[001-010],volta[001-003] # 2080 Tis only
##SBATCH --exclude=cuda[001-008],gpu[005-014],pascal[001-010],volta[001-003] # 3090 or A100
##SBATCH --exclude=cuda[001-008],gpu[005-006],gpu[015-016],gpu[019-026],pascal[001-010],volta[001-003]  # 2080 Ti or 3090

source /scratch/mas1107/nanoGPT/env/bin/activate
python -u train.py config/train_gpt2.py --wandb_project=monosemantic --n_layer=1 --n_head=512
deactivate
