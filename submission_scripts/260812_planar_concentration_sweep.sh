#!/bin/bash
#SBATCH --account=b53010
#SBATCH --partition=buyin
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=260812_planar_concentration_sweep
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryan.neff@northwestern.edu

set -euo pipefail

# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

# SensorModeling repository. This directory must contain protocol_scripts/
# and the utils package.
PROJECT_DIR="/home/rnt2664/SensorModeling"

# Use the full path so the batch job activates the intended environment.
CONDA_ENV="/home/rnt2664/.conda/envs/sensor-modeling-env"

PYTHON_SCRIPT="${PROJECT_DIR}/protocol_scripts/planar_concentration_sweep.py"

# The existing Params JSON can be reused if it contains only Params fields;
# concentration and geometry are overridden below by the protocol script.
PARAMS_JSON="${PROJECT_DIR}/configs/soloNanopore_base_params.json"

OUTPUT_ROOT="${PROJECT_DIR}/results/260812_planar_concentration_sweep"

# -----------------------------------------------------------------------------
# Protocol settings
# -----------------------------------------------------------------------------

# Association concentrations, in molar units.
CONCENTRATIONS_M=(
    10e-12
    100e-12
    1e-9
    10e-9
    100e-9
)

N_REPLICATES=5

ASSOCIATION_S=600
DISSOCIATION_S=1200

# Ligand-free wash.
DISSOCIATION_CONCENTRATION_M=0

# Planar surface height. Zero reproduces the standard flat geometry.
SURFACE_Z_M=0

RECORD_EVERY_S=0.05
ASSOCIATION_FRAMES=20
DISSOCIATION_FRAMES=20
TABLE_FORMAT="parquet"

# Set to true to replace existing concentration/replicate directories.
OVERWRITE=true

N_WORKERS=16

# -----------------------------------------------------------------------------
# Quest software environment
# -----------------------------------------------------------------------------
unset PYTHONPATH
unset PYTHONHOME

module purge
module load anaconda3/2022.05

# Quest batch-job activation sequence for the conda module.
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# Keep each IPyParallel engine single-threaded internally.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${PROJECT_DIR}"

cd "${PROJECT_DIR}"

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

echo "============================================================"
echo "IPyParallel workers: ${N_WORKERS}"
echo "Project:             ${PROJECT_DIR}"
echo "Python script:       ${PYTHON_SCRIPT}"
echo "Params JSON:         ${PARAMS_JSON}"
echo "Output:              ${OUTPUT_ROOT}"
echo "Concentrations (M):  ${CONCENTRATIONS_M[*]}"
echo "============================================================"

date
echo "Python executable: $(which python)"
python --version

# Fail early if the wrong Python installation leaks into sys.path.
python - <<'PY'
import sys
import zmq
import ipyparallel

print("Python:", sys.executable)
print("zmq:", zmq.__file__)
print("ipyparallel:", ipyparallel.__file__)

bad_paths = [
    path for path in sys.path
    if "anaconda3/2022.05" in path
    or "/python3.9/site-packages" in path
]

if bad_paths:
    raise RuntimeError(
        "Incompatible Python paths detected: " + repr(bad_paths)
    )

print("Python environment check passed.")
PY

# -----------------------------------------------------------------------------
# Build optional command-line arguments
# -----------------------------------------------------------------------------

OPTIONAL_ARGS=()

if [[ "${OVERWRITE}" == "true" ]]; then
    OPTIONAL_ARGS+=(--overwrite)
fi

# -----------------------------------------------------------------------------
# Run the protocol
# -----------------------------------------------------------------------------

echo "Starting planar concentration sweep."
date

python -u "${PYTHON_SCRIPT}" \
    --params-json "${PARAMS_JSON}" \
    --concentrations-M "${CONCENTRATIONS_M[@]}" \
    --n-replicates "${N_REPLICATES}" \
    --n-workers "${N_WORKERS}" \
    --association-s "${ASSOCIATION_S}" \
    --dissociation-s "${DISSOCIATION_S}" \
    --dissociation-concentration-M "${DISSOCIATION_CONCENTRATION_M}" \
    --surface-z-m "${SURFACE_Z_M}" \
    --record-every-s "${RECORD_EVERY_S}" \
    --association-frames "${ASSOCIATION_FRAMES}" \
    --dissociation-frames "${DISSOCIATION_FRAMES}" \
    --table-format "${TABLE_FORMAT}" \
    --output-root "${OUTPUT_ROOT}" \
    "${OPTIONAL_ARGS[@]}"

echo "Protocol completed successfully."
date
