#!/bin/bash
#SBATCH --account=b53010
#SBATCH --partition=buyin
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=260922_escape_assay_curvature_sweep
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryan.neff@northwestern.edu

set -euo pipefail

PROJECT_DIR="/home/rnt2664/SensorModeling"
CONDA_ENV="sensor-modeling-env"

PYTHON_SCRIPT="${PROJECT_DIR}/protocol_scripts/escape_assay_curvature_sweep.py"
PARAMS_JSON="${PROJECT_DIR}/configs/escape_assay_base_params.json"
GEOMETRY_JSON="${PROJECT_DIR}/configs/escape_assay_curvature_sweep_geometry.json"
OUTPUT_ROOT="${PROJECT_DIR}/results/260922_escape_assay_curvature_sweep"

N_REPLICATES=5
# Keep this equal to #SBATCH --ntasks. There are 6 radii x 5 replicates.
N_WORKERS=30

unset PYTHONPATH
unset PYTHONHOME

module purge
module load anaconda3/2022.05

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}"

echo "============================================================"
echo "Spherical-bowl escape-assay curvature sweep"
echo "============================================================"
echo "Project:          ${PROJECT_DIR}"
echo "Python script:    ${PYTHON_SCRIPT}"
echo "Base parameters:  ${PARAMS_JSON}"
echo "Geometry sweep:   ${GEOMETRY_JSON}"
echo "Output:           ${OUTPUT_ROOT}"
echo "Replicates:       ${N_REPLICATES}"
echo "Workers:          ${N_WORKERS}"
echo "============================================================"

date
which python
python --version

python -u "${PYTHON_SCRIPT}" \
    --params-json "${PARAMS_JSON}" \
    --geometry-json "${GEOMETRY_JSON}" \
    --n-replicates "${N_REPLICATES}" \
    --n-workers "${N_WORKERS}" \
    --output-root "${OUTPUT_ROOT}"

echo "Protocol completed successfully."
date
