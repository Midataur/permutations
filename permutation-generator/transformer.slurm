#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Mon Jan 08 2024 21:45:49 GMT+1100 (Australian Eastern Daylight Time)

# Partition for the job:
#SBATCH --partition=gpu-a100

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="permutations-transformer-training"

# The project ID which this job should run under:
#SBATCH --account="punim2163"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1
# The amount of memory in megabytes per node:
#SBATCH --mem=64000

# Use this email address:
#SBATCH --mail-user=mpetschack@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=7-0:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:

# The job command(s):
source ./venv/bin/activate
python ./main.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s