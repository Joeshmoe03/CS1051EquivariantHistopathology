#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G         # Memory allocation per CPU
#SBATCH --cpus-per-task=3         # Number of CPU cores per task
#SBATCH --time=8:00:00            # Job time limit
#SBATCH --partition=gpu-standard     # Specify the GPU partition
#SBATCH --job-name=equivariantCell
#SBATCH --output=./logs/non-equivariant.log

source /home/jliem/myenv/bin/activate

cd /home/jliem/CS1051/DeepLearning_Project
pwd

# Run the training script
python ./trainEquivariantCell.py -nepoch 100 -lr 0.001 -batchSize 5