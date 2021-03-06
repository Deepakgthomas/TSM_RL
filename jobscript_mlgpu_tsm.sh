#!/bin/bash
#SBATCH --nodes=1 # request one node
#SBATCH --cpus-per-task=1  # ask for 1 cpu
#SBATCH --mem=16G # Maximum amount of memory this job will be given, try to estimate this to the best of your ability. This asks for 8 GB of ram.
#SBATCH --time=0-09:30:00 # ask that the job be allowed to run for 30 minutes.
#SBATCH --gres=gpu:1 #If you just need one gpu, you're done, if you need more you can change the number
#SBATCH --partition=gpu #specify the gpu partition

# everything below this line is optional, but are nice to have quality of life things
#SBATCH --output=tsm.%J.out # tell it to store the output console text to a file called job.<assigned job number>.out
#SBATCH --error=tsm.%J.err # tell it to store the error messages from the program (if it doesn't write them to normal console output) to a file called job.<assigned job muber>.err
#SBATCH --job-name="tsm" # a nice readable name to give your job so you know what it is when you see it in the queue, instead of just numbers

# under this we just do what we would normally do to run the program, everything above this line is used by slurm to tell it what your job needs for resources
# let's load the modules we need to do what we're going to do

module load ml-gpu/20210730
# let's make sure we're where we expect to be in the filesystem tree (my working directory is specified here)
cd /work/LAS/jannesar-lab/deepak/TSM_Working



# the commands we're running are below, this executes my python code
ml-gpu python3 4.prioritized_dqn_tsm.py


