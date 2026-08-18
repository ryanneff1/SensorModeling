#!/bin/bash
#SBATCH --account=b53010
#SBATCH --partition=buyin
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=260818_nanopore_concentration_sweep_wm
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryan.neff@northwestern.edu

set -euo pipefail

PROJECT_DIR="/home/rnt2664/SensorModeling"
CONDA_ENV="sensor-modeling-env"
PYTHON_SCRIPT="${PROJECT_DIR}/protocol_scripts/run_nanopore_concentration_sweep_wm.py"
PARAMS_JSON="${PROJECT_DIR}/configs/nanoporous_base_params.json"
OUTPUT_ROOT="${PROJECT_DIR}/results/260818_nanopore_concentration_sweep_wm"

CONCENTRATIONS_M=(100e-12 500e-12 1e-9 10e-9 100e-9)

DIAMETER_NM=40
HEIGHT_NM=100
PITCH_NM=60
RIM_Z_NM=100
LAYOUT="square"
EDGE_MARGIN_NM=""

N_REPLICATES=5
ASSOCIATION_S=300
DISSOCIATION_S=300
DISSOCIATION_CONCENTRATION_M=0
RESERVOIR_OFFSET_LAYERS=1
RECORD_EVERY_S=0.05
ASSOCIATION_FRAMES=20
DISSOCIATION_FRAMES=20
TABLE_FORMAT="parquet"

# Manually keep this equal to #SBATCH --ntasks.
N_WORKERS=16

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

echo "Concentrations (M): ${CONCENTRATIONS_M[*]}"
echo "Geometry (nm): diameter=${DIAMETER_NM}, height=${HEIGHT_NM}, pitch=${PITCH_NM}, rim=${RIM_Z_NM}"
echo "Replicates: ${N_REPLICATES}"
echo "Workers: ${N_WORKERS}"
echo "Output: ${OUTPUT_ROOT}"
date
which python
python --version

EXTRA_ARGS=()
if [[ -n "${EDGE_MARGIN_NM}" ]]; then
    EXTRA_ARGS+=(--edge-margin-nm "${EDGE_MARGIN_NM}")
fi

python -u "${PYTHON_SCRIPT}" \
    --params-json "${PARAMS_JSON}" \
    --concentrations-M "${CONCENTRATIONS_M[@]}" \
    --diameter-nm "${DIAMETER_NM}" \
    --height-nm "${HEIGHT_NM}" \
    --pitch-nm "${PITCH_NM}" \
    --rim-z-nm "${RIM_Z_NM}" \
    --layout "${LAYOUT}" \
    --n-replicates "${N_REPLICATES}" \
    --n-workers "${N_WORKERS}" \
    --association-s "${ASSOCIATION_S}" \
    --dissociation-s "${DISSOCIATION_S}" \
    --dissociation-concentration-M "${DISSOCIATION_CONCENTRATION_M}" \
    --reservoir-offset-layers "${RESERVOIR_OFFSET_LAYERS}" \
    --record-every-s "${RECORD_EVERY_S}" \
    --association-frames "${ASSOCIATION_FRAMES}" \
    --dissociation-frames "${DISSOCIATION_FRAMES}" \
    --table-format "${TABLE_FORMAT}" \
    --output-root "${OUTPUT_ROOT}" \
    --overwrite \
    "${EXTRA_ARGS[@]}"

echo "Protocol completed successfully."
date