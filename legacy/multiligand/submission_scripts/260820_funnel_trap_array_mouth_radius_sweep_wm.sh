#!/bin/bash
#SBATCH --account=b53010
#SBATCH --partition=buyin
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=260820_funnel_trap_array_mouth_radius_sweep
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryan.neff@northwestern.edu

set -euo pipefail

PROJECT_DIR="/home/rnt2664/SensorModeling"
CONDA_ENV="sensor-modeling-env"

PYTHON_SCRIPT="${PROJECT_DIR}/protocol_scripts/funnel_trap_array_parameter_sweep_wm.py"
PARAMS_JSON="${PROJECT_DIR}/configs/funnel_trap_array_base_params.json"
GEOMETRY_JSON="${PROJECT_DIR}/configs/funnel_trap_array_mouth_radius_sweep_geometry.json"
OUTPUT_ROOT="${PROJECT_DIR}/results/260820_funnel_trap_array_mouth_radius_sweep"

N_REPLICATES=5

ASSOCIATION_S=300
DISSOCIATION_S=300

ASSOCIATION_CONCENTRATION_M=10e-9
DISSOCIATION_CONCENTRATION_M=0

RESERVOIR_OFFSET_LAYERS=1

RECORD_EVERY_S=0.05
ASSOCIATION_FRAMES=20
DISSOCIATION_FRAMES=20
TABLE_FORMAT="parquet"

# Manually keep this equal to "#SBATCH --ntasks".
N_WORKERS=32

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
echo "Funnel-trap-array sweep with well-mixed reservoir"
echo "============================================================"
echo "Project:          ${PROJECT_DIR}"
echo "Python script:    ${PYTHON_SCRIPT}"
echo "Base parameters:  ${PARAMS_JSON}"
echo "Geometry config:  ${GEOMETRY_JSON}"
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
    --association-s "${ASSOCIATION_S}" \
    --dissociation-s "${DISSOCIATION_S}" \
    --association-concentration-M "${ASSOCIATION_CONCENTRATION_M}" \
    --dissociation-concentration-M "${DISSOCIATION_CONCENTRATION_M}" \
    --reservoir-offset-layers "${RESERVOIR_OFFSET_LAYERS}" \
    --record-every-s "${RECORD_EVERY_S}" \
    --association-frames "${ASSOCIATION_FRAMES}" \
    --dissociation-frames "${DISSOCIATION_FRAMES}" \
    --table-format "${TABLE_FORMAT}" \
    --output-root "${OUTPUT_ROOT}" \
    --overwrite

echo "Protocol completed successfully."
date
