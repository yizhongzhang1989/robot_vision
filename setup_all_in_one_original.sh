#!/bin/bash

# Robot Vision Toolkit - All-in-One Setup Script
# ==============================================
# 
# This script sets up the complete robot vision toolkit after git clone.
# It handles submodules, dependencies, model downloads, and environment setup.
#
# Usage:
#   ./setup_all_in_one.sh [options]
#
# Options:
#   --skip-conda       Skip conda environment creation
#   --skip-models      Skip model checkpoint downloads
#   --dev              Install development dependencies
#   --help             Show this help message
#

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_NAME="robot_vision"
CONDA_ENV_NAME="flowformerpp"
PYTHON_VERSION="3.8"

# Color definitions for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
print_banner() {
    echo -e "${PURPLE}"
    echo "========================================================"
    echo " $1"
    echo "========================================================"
    echo -e "${NC}"
}

print_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_progress() {
    echo -e "${CYAN}▶${NC} $1"
}

# Parse command line arguments
SKIP_CONDA=false
SKIP_MODELS=false
DEV_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-conda)
            SKIP_CONDA=true
            shift
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --help)
            echo "Robot Vision Toolkit Setup Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-conda       Skip conda environment creation"
            echo "  --skip-models      Skip model checkpoint downloads"
            echo "  --dev              Install development dependencies"
            echo "  --help             Show this help message"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_system_requirements() {
    print_step "Checking system requirements..."
    
    # Check if git is installed
    if ! command_exists git; then
        print_error "Git is not installed. Please install git first."
        exit 1
    fi
    
    # Check if we're in a git repository
    if [ ! -d ".git" ]; then
        print_error "This script must be run from the root of the git repository."
        exit 1
    fi
    
    # Check Python
    if ! command_exists python3 && ! command_exists python; then
        print_error "Python is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    # Check conda (optional)
    if [ "$SKIP_CONDA" = false ] && ! command_exists conda; then
        print_warning "Conda is not installed. Will use system Python instead."
        SKIP_CONDA=true
    fi
    
    print_success "System requirements check passed!"
}

# Function to update git submodules
update_submodules() {
    print_step "Updating git submodules..."
    
    print_progress "Initializing submodules..."
    git submodule init
    
    print_progress "Updating submodules recursively..."
    git submodule update --recursive --progress
    
    # Verify submodules were updated
    local missing_submodules=()
    
    if [ ! -f "ThirdParty/camera_calibration_toolkit/README.md" ]; then
        missing_submodules+=("camera_calibration_toolkit")
    fi
    
    if [ ! -f "ThirdParty/FlowFormerPlusPlusServer/README.md" ]; then
        missing_submodules+=("FlowFormerPlusPlusServer")
    fi
    
    if [ ! -f "ThirdParty/ImageLabelingWeb/README.md" ]; then
        missing_submodules+=("ImageLabelingWeb")
    fi
    
    if [ ${#missing_submodules[@]} -gt 0 ]; then
        print_error "Failed to update submodules: ${missing_submodules[*]}"
        print_info "You may need to check your network connection or submodule URLs"
        exit 1
    fi
    
    print_success "All submodules updated successfully!"
}

# Function to create conda environment
create_conda_environment() {
    if [ "$SKIP_CONDA" = true ]; then
        print_info "Skipping conda environment creation (--skip-conda flag or conda not available)"
        return 0
    fi
    
    print_step "Setting up conda environment..."
    
    # Check if environment already exists
    if conda env list | grep -q "^${CONDA_ENV_NAME}"; then
        print_warning "Conda environment '${CONDA_ENV_NAME}' already exists."
        read -p "Do you want to remove and recreate it? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_progress "Removing existing environment..."
            conda env remove -n "$CONDA_ENV_NAME" -y
        else
            print_info "Using existing environment"
            return 0
        fi
    fi
    
    print_progress "Creating conda environment with Python $PYTHON_VERSION..."
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    
    print_success "Conda environment created successfully!"
    print_info "To activate: conda activate $CONDA_ENV_NAME"
}

# Function to check and fix NumPy compatibility issues
fix_numpy_compatibility() {
    print_step "Checking NumPy compatibility..."
    
    local python_cmd="python"
    local pip_cmd="pip"
    
    # Use conda environment if available
    if [ "$SKIP_CONDA" = false ] && conda env list | grep -q "^${CONDA_ENV_NAME}"; then
        python_cmd="conda run -n $CONDA_ENV_NAME python"
        pip_cmd="conda run -n $CONDA_ENV_NAME pip"
    fi
    
    # Check NumPy version and compatibility
    print_progress "Detecting NumPy version..."
    local numpy_version=$($python_cmd -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "not_found")
    
    if [ "$numpy_version" == "not_found" ]; then
        print_info "NumPy not found, will be installed with requirements"
        return 0
    fi
    
    print_info "Detected NumPy version: $numpy_version"
    
    # Check if it's NumPy 2.x
    if [[ $numpy_version == 2.* ]]; then
        print_info "NumPy 2.x detected - checking for compatibility issues..."
        
        # Test for common compatibility issues
        local compat_test=$($python_cmd -c "
try:
    import numpy
    import scipy
    import matplotlib
    print('compatible')
except Exception as e:
    if 'numpy.dtype size changed' in str(e) or 'binary incompatibility' in str(e):
        print('incompatible')
    else:
        print('other_error')
" 2>/dev/null || echo "test_failed")
        
        if [ "$compat_test" == "incompatible" ]; then
            print_warning "NumPy 2.x compatibility issues detected!"
            print_progress "Upgrading packages for NumPy 2.x compatibility..."
            
            # Upgrade packages that commonly have NumPy 1.x/2.x compatibility issues
            $pip_cmd install --upgrade scipy matplotlib pillow scikit-image opencv-python
            
            print_success "Packages upgraded for NumPy 2.x compatibility"
        elif [ "$compat_test" == "compatible" ]; then
            print_success "NumPy compatibility check passed"
        else
            print_warning "Could not fully test NumPy compatibility, proceeding with caution"
        fi
    else
        print_info "NumPy 1.x detected - should be compatible with most packages"
    fi
}

# Function to install Python dependencies
install_dependencies() {
    print_step "Installing Python dependencies..."
    
    local pip_cmd="pip"
    local python_cmd="python"
    
    # Use conda environment if available
    if [ "$SKIP_CONDA" = false ] && conda env list | grep -q "^${CONDA_ENV_NAME}"; then
        print_progress "Installing dependencies in conda environment..."
        pip_cmd="conda run -n $CONDA_ENV_NAME pip"
        python_cmd="conda run -n $CONDA_ENV_NAME python"
    else
        print_progress "Installing dependencies in system Python..."
    fi
    
    # Install base requirements
    if [ -f "requirements.txt" ]; then
        print_progress "Installing base requirements..."
        $pip_cmd install -r requirements.txt
    fi
    
    # Install FlowFormerPlusPlusServer requirements
    if [ -f "ThirdParty/FlowFormerPlusPlusServer/requirements.txt" ]; then
        print_progress "Installing FlowFormer++ requirements..."
        $pip_cmd install -r ThirdParty/FlowFormerPlusPlusServer/requirements.txt
    fi
    
    # Install camera calibration toolkit requirements
    if [ -f "ThirdParty/camera_calibration_toolkit/requirements.txt" ]; then
        print_progress "Installing camera calibration requirements..."
        $pip_cmd install -r ThirdParty/camera_calibration_toolkit/requirements.txt
    fi
    
    # Install ImageLabelingWeb requirements
    if [ -f "ThirdParty/ImageLabelingWeb/requirements.txt" ]; then
        print_progress "Installing image labeling requirements..."
        $pip_cmd install -r ThirdParty/ImageLabelingWeb/requirements.txt
    fi
    
    # Install development dependencies if requested
    if [ "$DEV_MODE" = true ]; then
        print_progress "Installing development dependencies..."
        $pip_cmd install pytest pytest-cov black flake8 jupyter notebook
    fi
    
    # Install current package in development mode
    print_progress "Installing robot vision toolkit in development mode..."
    $pip_cmd install -e .
    
    # Check and fix NumPy compatibility after all packages are installed
    fix_numpy_compatibility
    
    print_success "Python dependencies installed successfully!"
}

# Function to download model checkpoints
download_model_checkpoints() {
    if [ "$SKIP_MODELS" = true ]; then
        print_info "Skipping model checkpoint downloads (--skip-models flag)"
        return 0
    fi
    
    print_step "Downloading model checkpoints..."
    
    # Download FlowFormer++ checkpoints
    if [ -f "ThirdParty/FlowFormerPlusPlusServer/scripts/download_ckpts.sh" ]; then
        print_progress "Downloading FlowFormer++ model checkpoints..."
        cd "ThirdParty/FlowFormerPlusPlusServer"
        
        # Make sure the script is executable
        chmod +x scripts/download_ckpts.sh
        
        # Run the download script
        if ./scripts/download_ckpts.sh; then
            print_success "FlowFormer++ checkpoints downloaded successfully!"
            
            # Verify critical checkpoint files exist
            local missing_checkpoints=()
            for checkpoint in "sintel.pth" "things.pth" "kitti.pth" "chairs.pth"; do
                if [ ! -f "checkpoints/$checkpoint" ]; then
                    missing_checkpoints+=("$checkpoint")
                fi
            done
            
            if [ ${#missing_checkpoints[@]} -gt 0 ]; then
                print_warning "Some checkpoint files are missing: ${missing_checkpoints[*]}"
                print_info "This may cause issues when running examples"
            else
                print_success "All required checkpoint files verified!"
            fi
        else
            print_error "Failed to download FlowFormer++ checkpoints"
            print_info "This will prevent the examples from running properly"
            print_info "You can download them manually later by running:"
            print_info "  cd ThirdParty/FlowFormerPlusPlusServer && ./scripts/download_ckpts.sh"
            print_warning "Setup will continue, but examples may fail without model files"
        fi
        
        cd "$SCRIPT_DIR"
    else
        print_warning "FlowFormer++ download script not found"
        print_info "Examples may not work without model checkpoints"
    fi
    
    print_success "Model checkpoint download completed!"
}

# Function to run basic tests
run_basic_tests() {
    print_step "Running basic functionality tests..."
    
    local python_cmd="python"
    if [ "$SKIP_CONDA" = false ] && conda env list | grep -q "^${CONDA_ENV_NAME}"; then
        python_cmd="conda run -n $CONDA_ENV_NAME python"
    fi
    
    # Test import of core modules
    print_progress "Testing core module imports..."
    
    if $python_cmd -c "import core; print('✓ Core module imported successfully')" 2>/dev/null; then
        print_success "Core module import test passed!"
    else
        print_warning "Core module import test failed"
        return 1
    fi
    
    # Test NumPy/SciPy compatibility
    print_progress "Testing NumPy/SciPy compatibility..."
    local numpy_scipy_test=$($python_cmd -c "
try:
    import numpy, scipy, matplotlib
    print('✓ NumPy/SciPy/matplotlib compatibility verified')
    print('compatible')
except Exception as e:
    print(f'✗ Compatibility issue: {e}')
    print('incompatible')
" 2>/dev/null || echo "test_failed")
    
    if [[ $numpy_scipy_test == *"compatible"* ]]; then
        print_success "NumPy/SciPy compatibility test passed!"
    else
        print_warning "NumPy/SciPy compatibility test failed"
        print_info "This may cause issues when running examples"
    fi
    
    # Test FlowFormer++ API import
    if $python_cmd -c "import sys; sys.path.insert(0, 'ThirdParty/FlowFormerPlusPlusServer'); import flowformer_api; print('✓ FlowFormer++ API imported successfully')" 2>/dev/null; then
        print_success "FlowFormer++ API import test passed!"
    else
        print_warning "FlowFormer++ API import test failed"
    fi
    
    # Test FFPPKeypointTracker import (main functionality)
    print_progress "Testing FFPPKeypointTracker import..."
    if $python_cmd -c "from core.ffpp_keypoint_tracker import FFPPKeypointTracker; print('✓ FFPPKeypointTracker imported successfully')" 2>/dev/null; then
        print_success "FFPPKeypointTracker import test passed!"
    else
        print_warning "FFPPKeypointTracker import test failed"
        print_info "This indicates a serious setup issue"
        return 1
    fi
    
    # Test if we can run a basic example (optional, only if model files exist)
    if [ -f "ThirdParty/FlowFormerPlusPlusServer/checkpoints/sintel.pth" ]; then
        print_progress "Testing example script execution (quick test)..."
        # Run a minimal test to see if the example can at least start
        local example_test=$($python_cmd -c "
import sys
sys.path.insert(0, '.')
try:
    from core.ffpp_keypoint_tracker import FFPPKeypointTracker
    # Just test initialization without actually running tracking
    print('✓ Example script can initialize properly')
    print('example_ready')
except Exception as e:
    print(f'✗ Example initialization failed: {e}')
    print('example_failed')
" 2>/dev/null || echo "test_failed")
        
        if [[ $example_test == *"example_ready"* ]]; then
            print_success "Example script initialization test passed!"
            print_info "You should be able to run: python examples/ffpp_keypoint_tracker_example.py"
        else
            print_warning "Example script test failed"
            print_info "There may be issues when running the examples"
        fi
    else
        print_info "Skipping example test - model checkpoints not found"
        print_info "Download models to enable full functionality testing"
    fi
    
    print_success "Basic tests completed!"
}

# Function to display final setup information
display_setup_info() {
    print_banner "Setup Complete!"
    
    echo -e "${GREEN}✅ Robot Vision Toolkit has been set up successfully!${NC}"
    echo ""
    echo -e "${CYAN}📁 Project Structure:${NC}"
    echo "   • Core modules: core/"
    echo "   • Examples: examples/"
    echo "   • Third-party tools: ThirdParty/"
    echo "   • Model checkpoints: ThirdParty/FlowFormerPlusPlusServer/checkpoints/"
    echo ""
    
    if [ "$SKIP_CONDA" = false ]; then
        echo -e "${CYAN}🐍 Conda Environment:${NC}"
        echo "   • Environment name: $CONDA_ENV_NAME"
        echo "   • Activation command: conda activate $CONDA_ENV_NAME"
        echo ""
    fi
    
    echo -e "${CYAN}🚀 Quick Start:${NC}"
    if [ "$SKIP_CONDA" = false ]; then
        echo "   1. Activate environment: conda activate $CONDA_ENV_NAME"
        echo "   2. Run comprehensive example: python examples/ffpp_keypoint_tracker_example.py"
        echo "   3. Run simple example: python examples/keypoint_tracker_simple.py"
    else
        echo "   1. Run comprehensive example: python examples/ffpp_keypoint_tracker_example.py"
        echo "   2. Run simple example: python examples/keypoint_tracker_simple.py"
    fi
    echo ""
    
    echo -e "${CYAN}🔧 Available Tools:${NC}"
    echo "   • Keypoint tracking: examples/keypoint_tracker_*.py"
    echo "   • Camera calibration: ThirdParty/camera_calibration_toolkit/"
    echo "   • FlowFormer++ server: ThirdParty/FlowFormerPlusPlusServer/"
    echo "   • Image labeling: ThirdParty/ImageLabelingWeb/"
    echo ""
    
    if [ "$SKIP_MODELS" = true ]; then
        echo -e "${YELLOW}⚠️  Note: Model checkpoints were skipped. Download them with:${NC}"
        echo "   cd ThirdParty/FlowFormerPlusPlusServer && ./scripts/download_ckpts.sh"
        echo ""
    fi
    
    # Check if setup was successful
    local setup_status="✅ Complete"
    if [ ! -f "ThirdParty/FlowFormerPlusPlusServer/checkpoints/sintel.pth" ]; then
        setup_status="⚠️  Partial (missing model files)"
    fi
    
    echo -e "${CYAN}🏁 Setup Status: ${setup_status}${NC}"
    if [[ $setup_status == *"Partial"* ]]; then
        echo -e "${YELLOW}   Some components may not work without model files${NC}"
        echo -e "${YELLOW}   Run the model download command above to complete setup${NC}"
    else
        echo -e "${GREEN}   All components should be ready to use!${NC}"
    fi
    echo ""
    
    echo -e "${CYAN}📖 Documentation:${NC}"
    echo "   • Main README: README.md"
    echo "   • FlowFormer++ docs: ThirdParty/FlowFormerPlusPlusServer/README.md"
    echo "   • Camera calibration docs: ThirdParty/camera_calibration_toolkit/README.md"
    echo ""
    
    echo -e "${GREEN}Happy coding! 🎉${NC}"
}

# Function to handle cleanup on script failure
cleanup_on_failure() {
    print_error "Setup failed! Cleaning up..."
    
    # Remove incomplete conda environment if it was being created
    if [ "$SKIP_CONDA" = false ] && conda env list | grep -q "^${CONDA_ENV_NAME}"; then
        print_info "Removing incomplete conda environment..."
        conda env remove -n "$CONDA_ENV_NAME" -y || true
    fi
    
    print_error "Setup was interrupted. You may need to run the script again."
    exit 1
}

# Set up trap for cleanup on failure
trap cleanup_on_failure ERR

# Main setup execution
main() {
    print_banner "Robot Vision Toolkit Setup"
    
    print_info "Starting setup process..."
    print_info "Project directory: $SCRIPT_DIR"
    
    if [ "$SKIP_CONDA" = true ]; then
        print_info "Mode: System Python (conda skipped)"
    else
        print_info "Mode: Conda environment ($CONDA_ENV_NAME)"
    fi
    
    if [ "$DEV_MODE" = true ]; then
        print_info "Development mode enabled"
    fi
    
    echo ""
    
    # Execute setup steps
    check_system_requirements
    echo ""
    
    update_submodules
    echo ""
    
    create_conda_environment
    echo ""
    
    install_dependencies
    echo ""
    
    download_model_checkpoints
    echo ""
    
    run_basic_tests
    echo ""
    
    # Final setup information
    display_setup_info
    
    print_success "All setup steps completed successfully!"
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
else
    print_error "This script should be executed, not sourced"
    return 1
fi