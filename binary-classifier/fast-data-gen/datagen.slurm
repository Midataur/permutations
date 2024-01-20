#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Sun Jan 14 2024 18:38:25 GMT+1100 (Australian Eastern Daylight Time)

# Partition for the job:
#SBATCH --partition=cascade

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="fast-data-gen"

# The project ID which this job should run under:
#SBATCH --account="punim2163"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# The amount of memory in megabytes per node:
#SBATCH --mem=4096

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
#SBATCH --time=0-2:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)
module load GCC/11.3.0
module load Rust/1.65.0

# The job command(s):
cargo run --release -- -g 10 -m 45 -d 2000000 -t 10 -f ./data/forcegrok/train_data1.csv

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s