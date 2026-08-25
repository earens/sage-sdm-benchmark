#!/bin/bash
#SBATCH --job-name=sklearn_eval     # Job name
#SBATCH --cpus-per-task=4           # CPUs per job
#SBATCH --mem=24G                   # Memory per job
#SBATCH --time=12:00:00             # Max runtime
#SBATCH --output=logs/sklearn_eval_%A_%a.out
#SBATCH --error=logs/sklearn_eval_%A_%a.err

# Submit as a SLURM array from the repo root; adjust the SBATCH resources
# (and add --partition/--account) for your cluster:
#   sbatch --array=0-11 scripts/sklearn_eval.sh

SPECIES_PER_JOB=500
TOTAL_SPECIES=5771

# Activate the environment you installed — edit for your cluster, e.g.:
#   module load miniforge3 && source ~/.bashrc && conda activate sage

START=$((SLURM_ARRAY_TASK_ID * SPECIES_PER_JOB))
END=$((START + SPECIES_PER_JOB - 1))

if [ $END -ge $TOTAL_SPECIES ]; then
    END=$((TOTAL_SPECIES - 1))
fi

echo "Evaluating species $START to $END on $SLURM_JOB_NODELIST"

python scripts/traditional_sdms/sklearn_eval.py $START $END
