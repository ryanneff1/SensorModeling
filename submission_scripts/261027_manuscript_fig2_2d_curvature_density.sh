#!/bin/bash
#SBATCH --account=b53010
#SBATCH --partition=buyin
#SBATCH --job-name=fig2_2d_full
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --mem-per-cpu=2G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryan.neff@northwestern.edu

# 49 unique conditions x 8 replicates = 392 total tasks.
# With 30 workers, the allocation completes in approximately 14 task rounds.

set -euo pipefail

PROJECT_DIR=/home/rnt2664/SensorModeling
CONDA_ENV=sensor-modeling-env
N_WORKERS=30
OUTPUT_NAME=261027_manuscript_fig2_2d_curvature_density

unset PYTHONPATH PYTHONHOME
module purge
module load anaconda3/2022.05
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$PROJECT_DIR/results/$OUTPUT_NAME"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

python -u protocol_scripts/escape_assay_2d_manuscript_sweep.py \
  --params-json configs/manuscript_fig2_2d_base_params.json \
  --experiment-json configs/manuscript_fig2_2d_curvature_density.json \
  --n-workers "$N_WORKERS" \
  --output-root "results/$OUTPUT_NAME"
