#!/bin/bash

# Robot Vision Toolkit - Complete Setup Script
# ============================================
# Simple sequential setup script with Conda support
# 
# Usage:
#   bash setup_all_in_one.sh              # Use Conda (RECOMMENDED)
#   bash setup_all_in_one.sh --skip-conda # Use system Python (NOT RECOMMENDED)

set -e  # Exit on any error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"

# Parse command line arguments
SKIP_CONDA=false
SKIP_CONDA_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-conda)
            SKIP_CONDA=true
            SKIP_CONDA_FLAG="--skip-conda"
            shift
            ;;
        -h|--help)
            echo "Robot Vision Toolkit - Complete Setup Script"
            echo ""
            echo "Usage:"
            echo "  bash setup_all_in_one.sh              # Use Conda (RECOMMENDED)"
            echo "  bash setup_all_in_one.sh --skip-conda # Use system Python (NOT RECOMMENDED)"
            echo ""
            echo "Options:"
            echo "  --skip-conda    Skip Conda environment setup and use system Python"
            echo "  -h, --help      Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=================================="
echo "Robot Vision Toolkit - Setup"
echo "=================================="
echo ""
if [ "$SKIP_CONDA" = true ]; then
    echo "⚠️  WARNING: Running setup WITHOUT Conda environment!"
    echo "    This is NOT RECOMMENDED and may cause dependency conflicts."
    echo ""
fi
echo "Starting complete setup..."
echo ""

# Determine total steps based on whether conda is skipped
if [ "$SKIP_CONDA" = true ]; then
    TOTAL_STEPS=5
else
    TOTAL_STEPS=7
fi

# Step 1: Check system requirements
echo "Step 1/$TOTAL_STEPS: Checking system requirements..."
bash "$SCRIPTS_DIR/check_requirements.sh" $SKIP_CONDA_FLAG

# Step 2: Setup Git submodules
echo ""
echo "Step 2/$TOTAL_STEPS: Setting up Git submodules..."
bash "$SCRIPTS_DIR/setup_submodules.sh" update

# Step 3: Setup Conda environment (skip if --skip-conda is set)
if [ "$SKIP_CONDA" = false ]; then
    echo ""
    echo "Step 3/$TOTAL_STEPS: Setting up Conda environment..."
    bash "$SCRIPTS_DIR/setup_conda.sh" create
    
    echo ""
    echo "Step 4/$TOTAL_STEPS: Installing PyTorch and torchvision for Jetson..."
    bash "$SCRIPTS_DIR/setup_conda.sh" install-pytorch
    CURRENT_STEP=5
else
    echo ""
    echo "Step 3/$TOTAL_STEPS: Skipping Conda environment setup (--skip-conda flag provided)"
    CURRENT_STEP=3
fi

# Step 5/3: Install dependencies
echo ""
echo "Step $CURRENT_STEP/$TOTAL_STEPS: Installing dependencies..."
bash "$SCRIPTS_DIR/install_dependencies.sh" install $SKIP_CONDA_FLAG
CURRENT_STEP=$((CURRENT_STEP + 1))

# Step 5/4: Download models
echo ""
echo "Step $CURRENT_STEP/$TOTAL_STEPS: Downloading models..."
bash "$SCRIPTS_DIR/download_models.sh" download $SKIP_CONDA_FLAG
CURRENT_STEP=$((CURRENT_STEP + 1))

# Step 6/5: Run tests
echo ""
echo "Step $CURRENT_STEP/$TOTAL_STEPS: Running tests..."
bash "$SCRIPTS_DIR/run_tests.sh" all $SKIP_CONDA_FLAG

echo ""
echo "=================================="
echo "✅ Setup completed successfully!"
echo "=================================="
echo ""
if [ "$SKIP_CONDA" = false ]; then
    echo "Before running examples, activate the Conda environment:"
    echo "  conda activate robot_vision"
    echo ""
fi
echo "You can now:"
if [ "$SKIP_CONDA" = false ]; then
    echo "  • Activate environment: conda activate robot_vision"
fi
echo "  • Run examples: cd examples && python keypoint_tracker_simple.py"
echo "  • Start web services: python start_services.py"
echo "  • Test installation: scripts/run_tests.sh quick"
echo ""
