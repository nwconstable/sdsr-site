#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --mem=32GB
#SBATCH --tmp=4GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=const112@umn.edu
#SBATCH -p msismall
#SBATCH -o testrun_%j.output
#SBATCH -e testrun_%j.error

module load python/3.13

