#!/bin/bash
#SBATCH --account=asccasc
#SBATCH --nodes=1
#SBATCH --partition=pbatch
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/g/g91/ranganath2/cardiac-ai/logs/job-id-%j.txt

time=`date +%Y%m%d-%H%M%S`
mv /g/g91/ranganath2/cardiac-ai/logs/job-id-${SLURM_JOB_ID}.txt /g/g91/ranganath2/cardiac-ai/logs/${SLURM_JOB_ID}-${time}.txt

while [[ $# -gt 1 ]]
  do
    key="$1"

    case $key in
      -c|--config_path)
      CONFIG_PATH="$2"
      shift # past argument
      ;;
      -d|--save_dir)
      SAVE_DIR="$2"
      shift # past argument
      ;;
      -t|--tag)
      TAG="$2"
      shift # past argument
      ;;
      -r|--resume)
      TUNE="$2"
      shift
      ;;
      -rt|--ray_tune)
      RT="$2"
      shift
      ;;
      *) # unknown option
      ;;
    esac
  shift # past argument or value
  done

##### These are shell commands
module load cuda/10.2.89


# Collect arguments


export LD_LIBRARY_PATH=$HOME/.miniconda3/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/fast2:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0,1

ulimit -n 10000

srun --mpibind=off python -u train.py \
    --config_path=${CONFIG_PATH} \
    --tag=${TAG} \
    --save_dir=${SAVE_DIR} \
    --tune=${TUNE} \
    --writer="/g/g91/ranganath2/cardiac-ai/logs/${SLURM_JOB_ID}-${time}.txt"



# Remove all junk
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

echo 'Training Completed!'