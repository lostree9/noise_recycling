#!/bin/bash
#SBATCH --job-name=gj_paper
#SBATCH --output=logs/paper_%A_%a.out
#SBATCH --error=logs/paper_%A_%a.err
#SBATCH --array=0-54
#SBATCH --time=04:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

# ============================================================
# PRA paper: full geometry sweep
# Generate params first: python run_all.py --generate-params
# Then submit: sbatch submit_all.sh
# ============================================================

module load apptainer/latest

FENICS_SIF="${FENICS_SIF:-$HOME/new_thesus/dolfinx-stable.sif}"
CACHE="$HOME/.fenics_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$CACHE"

PROJECT_DIR="$HOME/new_start"
WORK_DIR="${PROJECT_DIR}/results/paper_${SLURM_ARRAY_JOB_ID}"
PARAMS_DIR="${PROJECT_DIR}/params"
CODE_DIR="${PROJECT_DIR}"

mkdir -p "$WORK_DIR" logs

PARAM_FILE="${PARAMS_DIR}/geom_${SLURM_ARRAY_TASK_ID}.json"
[ ! -f "$PARAM_FILE" ] && echo "Missing: $PARAM_FILE" && exit 1

echo "Task ${SLURM_ARRAY_TASK_ID}: $(cat $PARAM_FILE | python3 -c 'import sys,json; print(json.load(sys.stdin).get(\"name\",\"?\"))')"
cp "$PARAM_FILE" "${WORK_DIR}/"

apptainer exec \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    --env PYTHONPATH="${CODE_DIR}:${HOME}/new_start/local_packages:${PYTHONPATH}" \
    --env XDG_CACHE_HOME="${CACHE}" \
    "$FENICS_SIF" \
    bash -c "source /usr/local/bin/dolfinx-real-mode && python3 ${CODE_DIR}/run_single_general.py ${WORK_DIR}/geom_${SLURM_ARRAY_TASK_ID}.json"

echo "Task ${SLURM_ARRAY_TASK_ID} exit code: $?"
rm -rf "$CACHE"
