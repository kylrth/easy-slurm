#!/bin/bash
# remember to run setup.sh once first
#
#SBATCH --time=0-00:30:00  # adjust this to match the expected walltime of your job
#SBATCH --nodes=1          # use only one compute node per job
#SBATCH --ntasks=1         # for starting several processes in parallel from one job script. See https://stackoverflow.com/questions/39186698/what-does-the-ntasks-or-n-tasks-does-in-slurm
#SBATCH --gpus-per-node=1  # number of GPU devices to use in each compute node
#SBATCH --cpus-per-gpu=4   # number of CPU cores per allocated GPU
#SBATCH --mem-per-gpu=8G   # amount of CPU memory per GPU
#SBATCH --array=0-7        # submit 8 jobs, each with $SLURM_ARRAY_TASK_ID set to a number between 0-7 (inclusive)
#SBATCH --job-name=wilds_irm  # Optional. This changes the job's display name when you run the "squeue" command.
# #SBATCH --mail-type=ALL  # Optional. Receive emails when job begins, ends or fails.
# #SBATCH --mail-user=your_mail@umontreal.ca  # Optional. Where to receive all the job-related emails.
#
# While working on this script, it might be easiest to create an interactive job to run the commands
# yourself. You can do this by running `salloc --time=0-00:30:00 ...` (with all the same sbatch
# configurations as above).

set -e

# track walltime
date

# This time we load CUDA modules as well.
module load gcc arrow python/3.11 scipy-stack cuda cudnn

# TODO: change the directories below to the appropriate locations on your cluster
SOURCE=~/easy-slurm/wilds_irm  # location of this example code
PROJECT=~/projects/def-sponsor00/"$USER"/easy-slurm/wilds_irm  # location for project data, pre-trained models, etc.
SCRATCH=~/scratch/easy-slurm/wilds_irm  # location for experiment outputs

# $SLURM_TMPDIR is a temporary directory on a fast disk on the compute node. We bring data and
# environments to this directory for processing to be as fast as possible.
VENV="$SLURM_TMPDIR"/.venv
DATA_DIR="$SLURM_TMPDIR"/data

# set up the Python environment for this job
# This time, since CUDA is available, the CUDA-enabled version of PyTorch will be installed from the DRAC wheelhouse.
virtualenv --no-download "$VENV"
source "$VENV"/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r "$SOURCE"/requirements_slurm.txt
pip install --no-index "$PROJECT"/wheels/*

# copy our data from the project directory, extracting only the dataset needed for this experiment
# $SLURM_ARRAY_TASK_ID comes from the --array config at the top of this file.
if [ -z ${SLURM_ARRAY_TASK_ID+x} ]; then
    DATASET=""  # extract everything
elif [ "$SLURM_ARRAY_TASK_ID" -lt 4 ]; then
    DATASET="./waterbirds_v1.0"
else
    DATASET="./celebA_v1.0"
fi
mkdir -p "$DATA_DIR"
tar -xf "$PROJECT"/data.tar --directory "$DATA_DIR" "$DATASET"
echo "extracted $DATASET to $DATA_DIR"

date  # track time for setup

# We need to be able to import Python modules in the common/ directory, so we add to the PYTHONPATH.
cd "$SOURCE"/..  # project top level
PYTHONPATH="${PYTHONPATH}:." python -m wilds_irm $SLURM_ARRAY_TASK_ID \
    --wilds "$DATA_DIR" \
    --model_cache "$PROJECT"/torch_cache \
    --checkpoints "$SCRATCH"/checkpoints \
    --runs "$SCRATCH"/runs

date  # track time to finish
