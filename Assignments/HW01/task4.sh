#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J FirstSlurm
#SBATCH -o FirstSlurm.out -e FirstSlurm.err
#SBATCH -c 2

echo $HOSTNAME