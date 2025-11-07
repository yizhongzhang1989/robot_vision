#!/bin/bash

# Robot Vision Toolkit - Conda Environment Setup
# ===============================================
# Creates and manages conda environments

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/setup_utils.sh"

set -e  # Exit on any error

# Function to create conda environment
create_conda_environment() {
    local skip_conda=${1:-false}
    local force_recreate=${2:-false}
    
    if [ "$skip_conda" = true ]; then
        print_info "Skipping conda environment creation (--skip-conda flag)"
        return 0
    fi
    
    # Check if conda is available
    if ! command_exists conda; then
        print_warning "Conda is not installed. Cannot create conda environment."
        print_info "Install Miniconda or Anaconda to use conda environments."
        return 1
    fi
    
    print_step "Setting up conda environment..."
    
    # Check if environment already exists
    if conda_env_exists; then
        print_warning "Conda environment '${CONDA_ENV_NAME}' already exists."
        
        if [ "$force_recreate" = true ]; then
            print_progress "Force recreating environment..."
            remove_conda_environment
        else
            read -p "Do you want to remove and recreate it? [y/N]: " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                print_progress "Removing existing environment..."
                remove_conda_environment
            else
                print_info "Using existing environment"
                return 0
            fi
        fi
    fi
    
    print_progress "Creating conda environment with Python $PYTHON_VERSION and numpy<2..."
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" "numpy<2" -y
    
    # Verify environment was created
    if ! conda_env_exists; then
        print_error "Failed to create conda environment"
        return 1
    fi
    
    print_success "Conda environment created successfully!"
    print_info "Environment name: $CONDA_ENV_NAME"
    print_info "Python version: $PYTHON_VERSION"
    print_info "To activate: conda activate $CONDA_ENV_NAME"
}

# Function to install PyTorch and torchvision for Jetson
install_pytorch_jetson() {
    local skip_conda=${1:-false}
    
    if [ "$skip_conda" = true ]; then
        print_info "Skipping PyTorch Jetson installation (--skip-conda flag)"
        return 0
    fi
    
    if ! conda_env_exists; then
        print_error "Conda environment '$CONDA_ENV_NAME' does not exist"
        return 1
    fi
    
    print_step "Installing PyTorch and torchvision for Jetson..."
    
    # Install system dependencies for torchvision 
    # (checking if already installed to avoid unnecessarily prompting for password)
    print_progress "Checking system dependencies for torchvision..."
    local missing_deps=()
    for pkg in libjpeg-dev libpng-dev libtiff-dev; do
        if ! dpkg -l | grep -q "^ii  $pkg "; then
            missing_deps+=("$pkg")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_info "Need to install system packages: ${missing_deps[*]}"
        print_warning "This requires sudo access. You may be prompted for your password."
        if ! sudo apt-get install -y "${missing_deps[@]}"; then
            print_error "Failed to install system dependencies"
            return 1
        fi
        print_success "System dependencies installed"
    else
        print_success "All system dependencies already installed"
    fi
    
    # Upgrade pip and setuptools in conda environment
    print_progress "Upgrading pip and setuptools..."
    if ! conda run -n "$CONDA_ENV_NAME" pip install --upgrade pip setuptools 2>&1 | grep -E "(Successfully|Requirement already)" > /dev/null; then
        print_error "Failed to upgrade pip and setuptools"
        return 1
    fi
    
    # Ensure numpy<2 is installed before PyTorch
    print_progress "Ensuring numpy<2 is installed..."
    if ! conda run -n "$CONDA_ENV_NAME" pip install "numpy<2" 2>&1 | grep -E "(Successfully|Requirement already)" > /dev/null; then
        print_error "Failed to install numpy<2"
        return 1
    fi
    
    # Install PyTorch wheel for Jetson
    print_progress "Installing PyTorch from Jetson wheel..."
    local torch_wheel_url="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
    if ! conda run -n "$CONDA_ENV_NAME" pip install "$torch_wheel_url" 2>&1 | grep -E "(Successfully|Requirement already)" > /dev/null; then
        print_error "Failed to install PyTorch wheel"
        return 1
    fi
    
    # Clone and build torchvision from source
    local temp_dir=$(mktemp -d)
    local log_file="$PROJECT_ROOT/torchvision_build.log"
    
    print_progress "Cloning torchvision repository (logs: $log_file)..."
    cd "$temp_dir"
    
    if ! git clone https://github.com/pytorch/vision.git >> "$log_file" 2>&1; then
        print_error "Failed to clone torchvision repository"
        print_info "Check log file: $log_file"
        cd "$PROJECT_ROOT"
        rm -rf "$temp_dir"
        return 1
    fi
    
    cd vision
    print_progress "Checking out torchvision v0.20.0..."
    if ! git checkout tags/v0.20.0 >> "$log_file" 2>&1; then
        print_error "Failed to checkout v0.20.0"
        print_info "Check log file: $log_file"
        cd "$PROJECT_ROOT"
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Build and install torchvision with numpy<2 constraint
    print_progress "Building and installing torchvision (this may take 15+ minutes)..."
    print_info "Build logs: $log_file"
    
    # Create a temporary constraints file
    local constraints_file=$(mktemp)
    echo "numpy<2" > "$constraints_file"
    
    # Build and install with constraint (setup.py will respect PIP_CONSTRAINT env var)
    if ! conda run -n "$CONDA_ENV_NAME" bash -c "export PIP_CONSTRAINT='$constraints_file' && python3 setup.py install" >> "$log_file" 2>&1; then
        print_error "Failed to build and install torchvision"
        print_info "Check log file: $log_file"
        cd "$PROJECT_ROOT"
        rm -rf "$temp_dir"
        rm -f "$constraints_file"
        return 1
    fi
    print_success "Torchvision build completed!"
    
    # Install additional packages with numpy constraint
    print_progress "Installing Pillow, scipy, and matplotlib..."
    if ! conda run -n "$CONDA_ENV_NAME" pip install --constraint "$constraints_file" Pillow scipy matplotlib 2>&1 | grep -E "(Successfully|Requirement already)" > /dev/null; then
        print_error "Failed to install additional packages"
        cd "$PROJECT_ROOT"
        rm -rf "$temp_dir"
        rm -f "$constraints_file"
        return 1
    fi
    
    # Clean up constraints file
    rm -f "$constraints_file"
    
    # Clean up
    cd "$PROJECT_ROOT"
    rm -rf "$temp_dir"
    
    # Verify installation
    print_progress "Verifying PyTorch installation..."
    if conda run -n "$CONDA_ENV_NAME" python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"; then
        print_success "PyTorch and torchvision installed successfully!"
    else
        print_error "PyTorch verification failed"
        return 1
    fi
}

# Function to install conda packages
install_conda_packages() {
    local skip_conda=${1:-false}
    
    if [ "$skip_conda" = true ]; then
        print_info "Skipping conda package installation (--skip-conda flag)"
        return 0
    fi
    
    if ! command_exists conda; then
        print_error "Conda is not installed"
        return 1
    fi
    
    if ! conda_env_exists; then
        print_error "Conda environment '$CONDA_ENV_NAME' does not exist"
        print_info "Create it first with: $0 create"
        return 1
    fi
    
    print_step "Installing conda packages..."
    
    # Install packages needed for camera calibration toolkit (nlopt for optimization)
    print_progress "Installing optimization and scientific packages..."
    if conda install -n "$CONDA_ENV_NAME" nlopt -y; then
        print_success "Conda packages installed successfully!"
    else
        print_error "Failed to install conda packages"
        return 1
    fi
}

# Function to remove conda environment
remove_conda_environment() {
    if ! command_exists conda; then
        print_error "Conda is not installed"
        return 1
    fi
    
    if conda_env_exists; then
        print_step "Removing conda environment '$CONDA_ENV_NAME'..."
        conda env remove -n "$CONDA_ENV_NAME" -y
        print_success "Conda environment removed successfully!"
    else
        print_info "Conda environment '$CONDA_ENV_NAME' does not exist"
    fi
}

# Function to show environment info
show_environment_info() {
    print_step "Conda environment information..."
    
    if ! command_exists conda; then
        print_error "Conda is not installed"
        return 1
    fi
    
    echo -e "${CYAN}Conda environments:${NC}"
    conda env list
    
    if conda_env_exists; then
        echo -e "\n${CYAN}Environment details for '$CONDA_ENV_NAME':${NC}"
        conda list -n "$CONDA_ENV_NAME" | head -20
        echo "... (showing first 20 packages, use 'conda list -n $CONDA_ENV_NAME' for full list)"
        
        echo -e "\n${CYAN}Python version in environment:${NC}"
        conda run -n "$CONDA_ENV_NAME" python --version
    else
        print_warning "Environment '$CONDA_ENV_NAME' does not exist"
    fi
}

# Function to activate environment (for script use)
activate_environment() {
    if ! command_exists conda; then
        print_error "Conda is not installed"
        return 1
    fi
    
    if ! conda_env_exists; then
        print_error "Conda environment '$CONDA_ENV_NAME' does not exist"
        print_info "Create it first with: $0 create"
        return 1
    fi
    
    print_info "To activate the environment, run:"
    echo "    conda activate $CONDA_ENV_NAME"
    print_info "To deactivate later, run:"
    echo "    conda deactivate"
}

# Function to export environment
export_environment() {
    if ! conda_env_exists; then
        print_error "Conda environment '$CONDA_ENV_NAME' does not exist"
        return 1
    fi
    
    print_step "Exporting conda environment..."
    
    local export_file="$PROJECT_ROOT/environment.yml"
    conda env export -n "$CONDA_ENV_NAME" > "$export_file"
    
    print_success "Environment exported to: $export_file"
    print_info "To recreate this environment elsewhere, run:"
    echo "    conda env create -f environment.yml"
}

# Main execution
main() {
    local action=${1:-create}
    local skip_conda=${2:-false}
    local force_recreate=${3:-false}
    
    case "$action" in
        "create")
            print_banner "Conda Environment Setup"
            create_conda_environment "$skip_conda" "$force_recreate"
            ;;
        "install-pytorch")
            print_banner "PyTorch Jetson Installation"
            install_pytorch_jetson "$skip_conda"
            ;;
        "install-packages")
            print_banner "Conda Package Installation"
            install_conda_packages "$skip_conda"
            ;;
        "remove")
            print_banner "Conda Environment Removal"
            remove_conda_environment
            ;;
        "info")
            print_banner "Conda Environment Information"
            show_environment_info
            ;;
        "activate")
            activate_environment
            ;;
        "export")
            print_banner "Conda Environment Export"
            export_environment
            ;;
        *)
            print_error "Unknown action: $action"
            echo "Usage: $0 [create|install-pytorch|install-packages|remove|info|activate|export] [skip_conda] [force_recreate]"
            return 1
            ;;
    esac
}

# Execute if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi