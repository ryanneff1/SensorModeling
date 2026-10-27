#!/bin/bash
#SBATCH --account=b53010
#SBATCH --partition=buyin
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=fig5_sine_factorial
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryan.neff@northwestern.edu

# Total tasks: 3 amplitudes x 3 receptor densities x 51 occupancy states
#              x 8 replicates = 3,672.
# Censoring horizon: 1e6 / k_off = 1e8 s at k_off = 1e-2 s^-1.
# Existing 10,000 s manifests are detected as stale and recomputed.
set -euo pipefail
PROJECT_DIR="/home/rnt2664/SensorModeling"
CONDA_ENV="sensor-modeling-env"
N_WORKERS=30
OUTPUT_NAME="261016_manuscript_fig5_sinusoidal_factorial_response"
unset PYTHONPATH PYTHONHOME
module purge
module load anaconda3/2022.05
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${PROJECT_DIR}/results/${OUTPUT_NAME}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}"
python -u protocol_scripts/escape_assay_manuscript_sweep.py \
  --params-json configs/manuscript_escape_base_params.json \
  --experiment-json configs/manuscript_fig5_sinusoidal_factorial_response.json \
  --n-workers "${N_WORKERS}" \
  --output-root "results/${OUTPUT_NAME}"
