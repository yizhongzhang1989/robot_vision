#!/bin/bash

# Robot Vision Toolkit - Dependency Installation
# ==============================================
# Installs Python dependencies and handles compatibility

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/setup_utils.sh"

set -e  # Exit on any error

# Function to check and fix NumPy compatibility issues
fix_numpy_compatibility() {
    local skip_conda=${1:-false}
    
    print_step "Checking NumPy compatibility..."
    
    local python_cmd=$(get_python_cmd "$skip_conda")
    local pip_cmd=$(get_pip_cmd "$skip_conda")
    
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
    local skip_conda=${1:-false}
    local dev_mode=${2:-false}
    
    print_step "Installing Python dependencies..."
    
    local pip_cmd=$(get_pip_cmd "$skip_conda")
    local python_cmd=$(get_python_cmd "$skip_conda")
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Display installation context
    if [ "$skip_conda" = false ] && conda_env_exists; then
        print_progress "Installing dependencies in conda environment '$CONDA_ENV_NAME'..."
    else
        print_progress "Installing dependencies in system Python..."
    fi
    
    # Upgrade pip first
    print_progress "Upgrading pip..."
    $pip_cmd install --upgrade pip
    
    # Create a temporary constraints file to prevent numpy upgrade
    local constraints_file=$(mktemp)
    echo "numpy<2" > "$constraints_file"
    
    # Install base requirements
    if [ -f "requirements.txt" ]; then
        print_progress "Installing base requirements (with numpy<2 constraint)..."
        $pip_cmd install --constraint "$constraints_file" -r requirements.txt
    else
        print_warning "requirements.txt not found, skipping base requirements"
    fi
    
    # Install FlowFormerPlusPlusServer requirements
    local ffpp_req="ThirdParty/FlowFormerPlusPlusServer/requirements.txt"
    if [ -f "$ffpp_req" ]; then
        print_progress "Installing FlowFormer++ requirements (with numpy<2 constraint)..."
        $pip_cmd install --constraint "$constraints_file" -r "$ffpp_req"
    else
        print_warning "FlowFormer++ requirements.txt not found"
    fi
    
    # Install camera calibration toolkit requirements
    local calib_req="ThirdParty/camera_calibration_toolkit/requirements.txt"
    if [ -f "$calib_req" ]; then
        print_progress "Installing camera calibration requirements (with numpy<2 constraint)..."
        $pip_cmd install --constraint "$constraints_file" -r "$calib_req"
    else
        print_info "Camera calibration requirements.txt not found (optional)"
    fi
    
    # Install ImageLabelingWeb requirements
    local labeling_req="ThirdParty/ImageLabelingWeb/requirements.txt"
    if [ -f "$labeling_req" ]; then
        print_progress "Installing image labeling requirements (with numpy<2 constraint)..."
        $pip_cmd install --constraint "$constraints_file" -r "$labeling_req"
    else
        print_info "Image labeling requirements.txt not found (optional)"
    fi
    
    # Install development dependencies if requested
    if [ "$dev_mode" = true ]; then
        print_progress "Installing development dependencies..."
        $pip_cmd install --constraint "$constraints_file" pytest pytest-cov black flake8 jupyter notebook ipykernel
    fi
    
    # Install current package in development mode
    if [ -f "setup.py" ]; then
        print_progress "Installing robot vision toolkit in development mode..."
        $pip_cmd install --constraint "$constraints_file" -e .
    else
        print_warning "setup.py not found, skipping package installation"
    fi
    
    # Clean up constraints file
    rm -f "$constraints_file"
    
    # Check and fix NumPy compatibility after all packages are installed
    fix_numpy_compatibility "$skip_conda"
    
    print_success "Python dependencies installed successfully!"
}

# Function to install specific component dependencies
install_component_dependencies() {
    local component=$1
    local skip_conda=${2:-false}
    
    if [ -z "$component" ]; then
        print_error "Component name required"
        return 1
    fi
    
    print_step "Installing dependencies for component: $component"
    
    local pip_cmd=$(get_pip_cmd "$skip_conda")
    cd "$PROJECT_ROOT"
    
    case "$component" in
        "flowformer")
            local req_file="ThirdParty/FlowFormerPlusPlusServer/requirements.txt"
            ;;
        "calibration")
            local req_file="ThirdParty/camera_calibration_toolkit/requirements.txt"
            ;;
        "labeling")
            local req_file="ThirdParty/ImageLabelingWeb/requirements.txt"
            ;;
        "base")
            local req_file="requirements.txt"
            ;;
        *)
            print_error "Unknown component: $component"
            print_info "Available components: flowformer, calibration, labeling, base"
            return 1
            ;;
    esac
    
    if [ -f "$req_file" ]; then
        print_progress "Installing from $req_file..."
        $pip_cmd install -r "$req_file"
        print_success "Component dependencies installed successfully!"
    else
        print_error "Requirements file not found: $req_file"
        return 1
    fi
}

# Function to show dependency information
show_dependency_info() {
    local skip_conda=${1:-false}
    
    print_step "Dependency information..."
    
    local python_cmd=$(get_python_cmd "$skip_conda")
    local pip_cmd=$(get_pip_cmd "$skip_conda")
    
    echo -e "${CYAN}Python version:${NC}"
    $python_cmd --version
    
    echo -e "\n${CYAN}Pip version:${NC}"
    $pip_cmd --version
    
    echo -e "\n${CYAN}Key package versions:${NC}"
    $python_cmd -c "
import sys
packages = ['numpy', 'scipy', 'matplotlib', 'opencv-python', 'pillow', 'torch', 'torchvision']
for pkg in packages:
    try:
        mod = __import__(pkg.replace('-', '_'))
        version = getattr(mod, '__version__', 'unknown')
        print(f'  {pkg}: {version}')
    except ImportError:
        print(f'  {pkg}: not installed')
" 2>/dev/null || print_warning "Could not check package versions"
}

# Function to update all dependencies
update_dependencies() {
    local skip_conda=${1:-false}
    
    print_step "Updating all dependencies..."
    
    local pip_cmd=$(get_pip_cmd "$skip_conda")
    
    # Update pip first
    print_progress "Updating pip..."
    $pip_cmd install --upgrade pip
    
    # Update all installed packages
    print_progress "Updating all installed packages..."
    $pip_cmd list --outdated --format=freeze | cut -d= -f1 | xargs -r $pip_cmd install --upgrade
    
    print_success "Dependencies updated successfully!"
}

# Main execution
main() {
    local action=${1:-install}
    local skip_conda=false
    local dev_mode=false
    local component=""
    
    # Parse arguments
    shift  # Remove action
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-conda)
                skip_conda=true
                shift
                ;;
            --dev)
                dev_mode=true
                shift
                ;;
            --component)
                component="$2"
                shift 2
                ;;
            *)
                # For backward compatibility with positional args
                if [ -z "$component" ] && [ "$action" = "component" ]; then
                    component="$1"
                fi
                shift
                ;;
        esac
    done
    
    case "$action" in
        "install")
            print_banner "Dependency Installation"
            install_dependencies "$skip_conda" "$dev_mode"
            ;;
        "component")
            print_banner "Component Dependency Installation"
            install_component_dependencies "$component" "$skip_conda"
            ;;
        "info")
            print_banner "Dependency Information"
            show_dependency_info "$skip_conda"
            ;;
        "update")
            print_banner "Dependency Update"
            update_dependencies "$skip_conda"
            ;;
        "numpy-fix")
            print_banner "NumPy Compatibility Fix"
            fix_numpy_compatibility "$skip_conda"
            ;;
        *)
            print_error "Unknown action: $action"
            echo "Usage: $0 [install|component|info|update|numpy-fix] [--skip-conda] [--dev] [--component <name>]"
            return 1
            ;;
    esac
}

# Execute if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi