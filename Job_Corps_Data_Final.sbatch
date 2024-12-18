#!/bin/bash
#SBATCH --job-name job_corps_final    # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition medium         # Slurm partition to use
#SBATCH --ntasks 1              # Number of tasks to run. By default, one CPU core will be allocated per task
#SBATCH --time 3-11:10             # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=1600           # Memory limit for each tasks (in MB)
#SBATCH -o out/job_corps_final.out        # File to which STDOUT will be written
#SBATCH -e out/job_corps_final.err        # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu  # Email to which notifications will be sent

module load Python
source ../yikun_venv/bin/activate
python ./Job_Corps_Data_Final.py

