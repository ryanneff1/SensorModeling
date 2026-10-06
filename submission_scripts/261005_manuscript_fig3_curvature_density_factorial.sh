#!/bin/bash
#SBATCH --account=b53010
#SBATCH --partition=buyin
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=25
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=fig3_curvature_density
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryan.neff@northwestern.edu
set -euo pipefail
PROJECT_DIR="/home/rnt2664/SensorModeling"
CONDA_ENV="sensor-modeling-env"
CONFIG="manuscript_fig3_curvature_density_factorial.json"
OUTPUT_NAME="261005_manuscript_fig3_curvature_density_factorial"
N_WORKERS=25
unset PYTHONPATH PYTHONHOME
module purge
module load anaconda3/2022.05
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${PROJECT_DIR}/results/${OUTPUT_NAME}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}"
python -u protocol_scripts/escape_assay_manuscript_sweep.py --params-json configs/manuscript_escape_base_params.json --experiment-json "configs/${CONFIG}" --n-workers "${N_WORKERS}" --output-root "results/${OUTPUT_NAME}"
