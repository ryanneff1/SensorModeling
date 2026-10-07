#!/bin/bash
#SBATCH --account=b53010
#SBATCH --partition=buyin
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=final_lowC_self
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryan.neff@northwestern.edu

# Total tasks: 3 geometries x 21 background occupancies x 8 replicates = 504.
set -euo pipefail
PROJECT_DIR="/home/rnt2664/SensorModeling"
CONDA_ENV="sensor-modeling-env"
N_WORKERS=28
OUTPUT_NAME="261006_manuscript_final_low_concentration_self_only"
unset PYTHONPATH PYTHONHOME
module purge
module load anaconda3/2022.05
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${PROJECT_DIR}/results/${OUTPUT_NAME}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}"
python -u protocol_scripts/escape_assay_manuscript_sweep.py --params-json configs/manuscript_escape_base_params.json --experiment-json configs/manuscript_final_low_concentration_self_only.json --n-workers "${N_WORKERS}" --output-root "results/${OUTPUT_NAME}"
