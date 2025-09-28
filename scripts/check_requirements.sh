#!/bin/bash

# Robot Vision Toolkit - System Requirements Check
# ================================================
# Checks system requirements and prerequisites

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/setup_utils.sh"

set -e  # Exit on any error

# Function to check system requirements
check_system_requirements() {
    print_step "Checking system requirements..."
    
    # Check if git is installed
    if ! command_exists git; then
        print_error "Git is not installed. Please install git first."
        return 1
    fi
    
    # Check if we're in a git repository
    if [ ! -d "$PROJECT_ROOT/.git" ]; then
        print_error "This script must be run from the root of the git repository."
        print_info "Current directory: $(pwd)"
        print_info "Expected project root: $PROJECT_ROOT"
        return 1
    fi
    
    # Check Python
    if ! command_exists python3 && ! command_exists python; then
        print_error "Python is not installed. Please install Python 3.8+ first."
        return 1
    fi
    
    # Check Python version
    local python_cmd="python3"
    if ! command_exists python3; then
        python_cmd="python"
    fi
    
    local python_version=$($python_cmd --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
    local major_version=$(echo "$python_version" | cut -d. -f1)
    local minor_version=$(echo "$python_version" | cut -d. -f2)
    
    if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 8 ]); then
        print_error "Python 3.8+ is required. Found Python $python_version"
        return 1
    fi
    
    print_info "Python version: $python_version ✓"
    
    # Check conda (optional)
    local skip_conda=${1:-false}
    if [ "$skip_conda" = false ]; then
        if command_exists conda; then
            local conda_version=$(conda --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -1)
            print_info "Conda version: $conda_version ✓"
        else
            print_warning "Conda is not installed. Will use system Python instead."
            print_info "Consider installing Miniconda or Anaconda for better environment management."
        fi
    fi
    
    # Check available disk space
    local available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 5242880 ]; then  # 5GB in KB
        print_warning "Less than 5GB of disk space available. Model downloads may fail."
        print_info "Available space: $(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}')"
    else
        print_info "Disk space: $(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}') available ✓"
    fi
    
    # Check internet connectivity
    if ping -c 1 google.com >/dev/null 2>&1; then
        print_info "Internet connectivity: ✓"
    else
        print_warning "No internet connectivity detected. Downloads may fail."
    fi
    
    print_success "System requirements check completed!"
    return 0
}

# Main execution
main() {
    local skip_conda=${1:-false}
    
    print_banner "System Requirements Check"
    check_system_requirements "$skip_conda"
}

# Execute if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi