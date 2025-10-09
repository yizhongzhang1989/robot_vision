#!/bin/bash

# Robot Vision Toolkit - Complete Setup Script
# ============================================
# Simple sequential setup script
# To skip any step, just comment out the corresponding line

set -e  # Exit on any error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"

echo "=================================="
echo "Robot Vision Toolkit - Setup"
echo "=================================="
echo ""
echo "Starting complete setup..."
echo "To skip any step, comment it out in this script"
echo ""

# Step 1: Check system requirements
echo "Step 1/6: Checking system requirements..."
bash "$SCRIPTS_DIR/check_requirements.sh"

# Step 2: Setup Git submodules
echo ""
echo "Step 2/6: Setting up Git submodules..."
bash "$SCRIPTS_DIR/setup_submodules.sh" update

# Step 3: Setup Conda environment
echo ""
echo "Step 3/6: Setting up Conda environment..."
bash "$SCRIPTS_DIR/setup_conda.sh" create

# Step 4: Install dependencies
echo ""
echo "Step 4/6: Installing dependencies..."
bash "$SCRIPTS_DIR/install_dependencies.sh" install

# Step 5: Download models
echo ""
echo "Step 5/6: Downloading models..."
bash "$SCRIPTS_DIR/download_models.sh" download

# Step 6: Run tests
echo ""
echo "Step 6/6: Running tests..."
bash "$SCRIPTS_DIR/run_tests.sh" all

echo ""
echo "=================================="
echo "✅ Setup completed successfully!"
echo "=================================="
echo ""
echo "You can now:"
echo "  • Run examples: cd examples && python keypoint_tracker_simple.py"
echo "  • Start web services: python start_services.py"
echo "  • Test installation: scripts/run_tests.sh quick"
echo ""
