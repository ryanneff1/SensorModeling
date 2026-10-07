#!/bin/bash
#SBATCH --account=b53010
#SBATCH --partition=buyin
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=fig3_response_grid
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryan.neff@northwestern.edu

# Total tasks: 5 representative conditions x 21 occupancies x 8 replicates = 840.
set -euo pipefail
PROJECT_DIR="/home/rnt2664/SensorModeling"
CONDA_ENV="sensor-modeling-env"
N_WORKERS=30
OUTPUT_NAME="261006_manuscript_fig3_response_grid"
unset PYTHONPATH PYTHONHOME
module purge
module load anaconda3/2022.05
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${PROJECT_DIR}/results/${OUTPUT_NAME}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}"
python -u protocol_scripts/escape_assay_manuscript_sweep.py --params-json configs/manuscript_escape_base_params.json --experiment-json configs/manuscript_fig3_response_grid.json --n-workers "${N_WORKERS}" --output-root "results/${OUTPUT_NAME}"
