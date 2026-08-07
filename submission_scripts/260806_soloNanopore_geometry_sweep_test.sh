#!/bin/bash
#SBATCH --account=b53010
#SBATCH --partition=buyin
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=260806_nanopore_geometry_sweep
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryan.neff@northwestern.edu

set -euo pipefail

# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

# Full path to the SensorModeling repository. The directory must contain the
# protocol script and the utils package.
PROJECT_DIR="/Users/rynon/Documents/GitHub/SensorModeling"

# Full path to the Quest conda environment containing numpy, pandas, scipy,
# pyarrow, tqdm, ipyparallel, and the other packages used by the model.
CONDA_ENV="abmenv"

PYTHON_SCRIPT="${PROJECT_DIR}/protocol_scripts/nanopore_geometry_sweep.py"
PARAMS_JSON="${PROJECT_DIR}/configs/soloNanopore_base_params.json"
OUTPUT_ROOT="//Users/rynon/Documents/GitHub/SensorModeling/results/260806_nanopore_geometry_sweep_test"

# Protocol settings passed directly to the Python script.
DIAMETERS_M=(30e-9 35e-9 40e-9 50e-9 70e-9 90e-9)
N_REPLICATES=5

ASSOCIATION_S=0.01
DISSOCIATION_S=0.01
ASSOCIATION_CONCENTRATION_M=10e-9
DISSOCIATION_CONCENTRATION_M=0

PORE_DEPTH_M=250e-9
RIM_Z_M=250e-9
# PITCH_M=PORE_DEPTH_M * 1.2 # 20% larger than the pore depth to avoid overlap
LAYOUT="square"

RECORD_EVERY_S=0.05
ASSOCIATION_FRAMES=20
DISSOCIATION_FRAMES=20
TABLE_FORMAT="parquet"

# Leave empty to let the Python geometry function choose its default margin.
EDGE_MARGIN_M=""

# Set to true only when existing replicate directories should be replaced.
OVERWRITE=True

# Use one IPyParallel engine per allocated Slurm CPU. The Python script will
# automatically reduce this value if there are fewer tasks than engines.
N_WORKERS=4

# -----------------------------------------------------------------------------
# Quest software environment
# -----------------------------------------------------------------------------

# module purge
# module load anaconda3/2022.05

# Quest batch-job activation sequence for the conda module.
# conda activate "${CONDA_ENV}"

# Keep each IPyParallel engine single-threaded.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${OUTPUT_ROOT}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

echo "IPyParallel jobs: ${N_WORKERS}"
echo "Project:          ${PROJECT_DIR}"
echo "Output:           ${OUTPUT_ROOT}"
date
python --version

# -----------------------------------------------------------------------------
# Build optional command-line arguments
# -----------------------------------------------------------------------------

OPTIONAL_ARGS=()

if [[ -n "${EDGE_MARGIN_M}" ]]; then
    OPTIONAL_ARGS+=(--edge-margin-m "${EDGE_MARGIN_M}")
fi

if [[ "${OVERWRITE}" == "true" ]]; then
    OPTIONAL_ARGS+=(--overwrite)
fi

# -----------------------------------------------------------------------------
# Run the protocol
# -----------------------------------------------------------------------------

python -u "${PYTHON_SCRIPT}" \
    --params-json "${PARAMS_JSON}" \
    --diameters-m "${DIAMETERS_M[@]}" \
    --n-replicates "${N_REPLICATES}" \
    --n-workers "${N_WORKERS}" \
    --association-s "${ASSOCIATION_S}" \
    --dissociation-s "${DISSOCIATION_S}" \
    --association-concentration-M "${ASSOCIATION_CONCENTRATION_M}" \
    --dissociation-concentration-M "${DISSOCIATION_CONCENTRATION_M}" \
    --pore-depth-m "${PORE_DEPTH_M}" \
    --rim-z-m "${RIM_Z_M}" \
    --layout "${LAYOUT}" \
    --record-every-s "${RECORD_EVERY_S}" \
    --association-frames "${ASSOCIATION_FRAMES}" \
    --dissociation-frames "${DISSOCIATION_FRAMES}" \
    --table-format "${TABLE_FORMAT}" \
    --output-root "${OUTPUT_ROOT}" \

echo "Protocol completed successfully."
date
