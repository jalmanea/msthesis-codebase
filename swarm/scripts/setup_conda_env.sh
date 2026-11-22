#!/bin/bash
# ============================================================================
# Setup Conda Environment for DroneComm on HPC
# ============================================================================
#
# This script creates and configures the conda environment for running
# the drone deployment experiments on an HPC cluster.
#
# Usage:
#   # First time setup:
#   bash scripts/setup_conda_env.sh
#
#   # Activate the environment:
#   conda activate dronecomm
#
#   # Verify installation:
#   python -m pytest tests/
#
# ============================================================================

set -e

echo "=============================================="
echo "DroneComm Conda Environment Setup"
echo "=============================================="

# ----------------------------------------------------------------------------
# 1. Check for conda
# ----------------------------------------------------------------------------
echo ""
echo "1. Checking for conda..."

if ! command -v conda &> /dev/null; then
    echo "   ERROR: conda not found!"
    echo ""
    echo "   On KAUST Ibex, load the module first:"
    echo "     module load anaconda3"
    echo ""
    echo "   Or install miniconda:"
    echo "     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "     bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

echo "   Found: $(conda --version)"

# ----------------------------------------------------------------------------
# 2. Navigate to project directory
# ----------------------------------------------------------------------------
echo ""
echo "2. Navigating to project directory..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
echo "   Project dir: $PROJECT_DIR"

# ----------------------------------------------------------------------------
# 3. Create/update conda environment
# ----------------------------------------------------------------------------
echo ""
echo "3. Creating conda environment 'dronecomm'..."

if conda env list | grep -q "^dronecomm "; then
    echo "   Environment exists. Updating..."
    conda env update -f environment.yml --prune
else
    echo "   Creating new environment..."
    conda env create -f environment.yml
fi

# ----------------------------------------------------------------------------
# 4. Activate and verify
# ----------------------------------------------------------------------------
echo ""
echo "4. Verifying installation..."

# Source conda for script use
eval "$(conda shell.bash hook)"
conda activate dronecomm

echo "   Python: $(python --version)"
echo "   NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "   SciPy: $(python -c 'import scipy; print(scipy.__version__)')"
echo "   Matplotlib: $(python -c 'import matplotlib; print(matplotlib.__version__)')"

# Check dronecomm package
python -c "import dronecomm; print('   dronecomm: OK')"

# ----------------------------------------------------------------------------
# 5. Summary
# ----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "To use the environment:"
echo ""
echo "  # Activate:"
echo "  conda activate dronecomm"
echo ""
echo "  # Run tests:"
echo "  python -m pytest tests/"
echo ""
echo "  # Submit jobs:"
echo "  sbatch scripts/submit_heuristic_50.sh"
echo ""
