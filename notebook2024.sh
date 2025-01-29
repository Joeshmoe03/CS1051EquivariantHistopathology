#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10G         # Memory allocation per CPU
#SBATCH --cpus-per-task=3         # Number of CPU cores per task
#SBATCH --time=2:00:00            # Job time limit
#SBATCH --partition=gpu-short     # Specify the GPU partition
#SBATCH --job-name=jupyter-notebook
#SBATCH --output=./logs/jupyter-notebook.log

# Get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i 8000-9999 -n 1)
node=$(hostname -s)
user=$(whoami)

# Print tunneling instructions
echo -e "
==========================================================
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} jliem@ada.middlebury.edu


Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ada.middlebury.edu
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
https://${port}  (prefix w/ https:// if using password)
============================================================
"

# Load required modules or activate conda environment
# module load python/anaconda3
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate /home/jliem/.conda/envs/ASLenv
source /home/jliem/myenv/bin/activate

# DON'T USE ADDRESS BELOW.
# DO USE TOKEN BELOW
jupyter-notebook --no-browser --port=${port} --ip=${node}
