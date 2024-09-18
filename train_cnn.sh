#!/bin/bash
#SBATCH --nodes=1    # request only 1 node
#SBATCH --partition dept.appliedmath      # this job will be submitted to AM queue
#SBATCH --mem=64G #this job is asked for 64G of total memory, use 0 if you want to use entire node memory
#SBATCH --time=1-00:00:00 
#SBATCH --ntasks-per-node=56 # this job requests for 56 cores on a node
#SBATCH --output=my_%j.stdout    # standard output will be redirected to this file
# #SBATCH --constraint=bigmem   #uncomment this line if you need the access to the bigmem node for Pinnacles
# #SBATCH --constraint=gpu #uncomment this line if you need the access to GPU
#SBATCH --gres=gpu:1   #uncomment this line if you need GPU access (2 GPUs)
#SBATCH --job-name=cardiac_ai_cnn    # this is your job’s name
#SBATCH --mail-user=jornelasmunoz@ucmerced.edu  
#SBATCH --mail-type=ALL  #uncomment the first two lines if you want to receive the email notifications
#SBATCH --export=ALL
#  type 'man sbatch' for more information and options

# run your job
pwd; hostname; date
export PROJECT_DIR=/home/jornelasmunoz/cardiac-ai

module load anaconda3

source activate pytorch

echo 'Starting Matlab'
python3 -u $PROJECT_DIR/train_cnn.py

wait
pwd; hostname; date
echo 'Done'
