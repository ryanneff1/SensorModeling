#!/bin/bash
#SBATCH --account=b53010
#SBATCH --partition=buyin
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=260817_nanopore_height_sweep_well_mixed
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryan.neff@northwestern.edu

set -euo pipefail

# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

# Full path to the SensorModeling repository.
PROJECT_DIR="/home/rnt2664/SensorModeling"

# Quest conda environment.
CONDA_ENV="sensor-modeling-env"

PYTHON_SCRIPT="${PROJECT_DIR}/protocol_scripts/nanopore_height_sweep_wm.py"
PARAMS_JSON="${PROJECT_DIR}/configs/nanoporous_base_params.json"
OUTPUT_ROOT="${PROJECT_DIR}/results/260817_nanopore_height_sweep_wm"

# -----------------------------------------------------------------------------
# Nanopore geometry sweep
# -----------------------------------------------------------------------------

# Nanopore height is implemented as pore_depth_m in the geometry constructor:
# the distance from the upper pore rim to the pore bottom.
HEIGHTS_NM=(10 30 50 100 150 250)

# Fixed geometry parameters.
DIAMETER_NM=40
PITCH_NM=60
RIM_Z_NM=250
LAYOUT="square"

# Optional pore-center edge margin in nm.
# Leave empty to use make_nanopore_array_geometry()'s default.
EDGE_MARGIN_NM=""

# -----------------------------------------------------------------------------
# Replicates and simulation phases
# -----------------------------------------------------------------------------

N_REPLICATES=5

ASSOCIATION_S=300
DISSOCIATION_S=300

ASSOCIATION_CONCENTRATION_M=10e-9
DISSOCIATION_CONCENTRATION_M=0

# One explicitly diffusive lattice layer above the sensor envelope before
# reaching the internal well-mixed reservoir.
RESERVOIR_OFFSET_LAYERS=1

RECORD_EVERY_S=0.05
ASSOCIATION_FRAMES=20
DISSOCIATION_FRAMES=20
TABLE_FORMAT="parquet"

# Set this manually to match "#SBATCH --ntasks" above.
N_WORKERS=16

# -----------------------------------------------------------------------------
# Quest software environment
# -----------------------------------------------------------------------------

unset PYTHONPATH
unset PYTHONHOME

module purge
module load anaconda3/2022.05

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# Keep each IPyParallel worker single-threaded.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${OUTPUT_ROOT}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}"

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

echo "============================================================"
echo "Nanopore height sweep with well-mixed reservoir"
echo "============================================================"
echo "Project:             ${PROJECT_DIR}"
echo "Python script:       ${PYTHON_SCRIPT}"
echo "Parameters:          ${PARAMS_JSON}"
echo "Output:              ${OUTPUT_ROOT}"
echo "Heights (nm):        ${HEIGHTS_NM[*]}"
echo "Diameter (nm):       ${DIAMETER_NM}"
echo "Pitch (nm):          ${PITCH_NM}"
echo "Rim z (nm):          ${RIM_Z_NM}"
echo "Replicates:          ${N_REPLICATES}"
echo "IPyParallel workers: ${N_WORKERS}"
echo "Reservoir offset:    ${RESERVOIR_OFFSET_LAYERS} layer(s)"
echo "Association [M]:     ${ASSOCIATION_CONCENTRATION_M}"
echo "Dissociation [M]:    ${DISSOCIATION_CONCENTRATION_M}"
echo "============================================================"

date
which python
python --version

# -----------------------------------------------------------------------------
# Optional command-line arguments
# -----------------------------------------------------------------------------

EXTRA_ARGS=()

if [[ -n "${EDGE_MARGIN_NM}" ]]; then
    EXTRA_ARGS+=(
        --edge-margin-nm "${EDGE_MARGIN_NM}"
    )
fi

# -----------------------------------------------------------------------------
# Run the protocol
# -----------------------------------------------------------------------------

python -u "${PYTHON_SCRIPT}" \
    --params-json "${PARAMS_JSON}" \
    --heights-nm "${HEIGHTS_NM[@]}" \
    --diameter-nm "${DIAMETER_NM}" \
    --pitch-nm "${PITCH_NM}" \
    --rim-z-nm "${RIM_Z_NM}" \
    --layout "${LAYOUT}" \
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
    --overwrite \
    "${EXTRA_ARGS[@]}"

echo "Protocol completed successfully."
date