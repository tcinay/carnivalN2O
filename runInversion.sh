#!/bin/bash
#SBATCH --job-name=carnival_inversion
#SBATCH --output=logs/carnival_inversion_%a.log
#SBATCH --error=logs/carnival_inversion_%a.log
#SBATCH --array=0-1                   
#SBATCH -n 4
#SBATCH -c 4            

module load anaconda/2023b
source activate carnival_env

YEARS=(2024 2025)
YEAR="${YEARS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p logs outputs

/home/gridsan/tcinay/.conda/envs/carnival_env/bin/python inversions.py --year "$YEAR" --cores "$SLURM_CPUS_PER_TASK" --outdir outputs