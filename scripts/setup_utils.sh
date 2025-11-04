#!/bin/bash

# Robot Vision Toolkit - Setup Utilities
# =======================================
# Common utilities and functions used by setup scripts

# Script configuration
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
export PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PROJECT_NAME="robot_vision"
export CONDA_ENV_NAME="robot_vision"
export PYTHON_VERSION="3.8"

# Color definitions for output
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export BLUE='\033[0;34m'
export PURPLE='\033[0;35m'
export CYAN='\033[0;36m'
export NC='\033[0m' # No Color

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

print_step_number() {
    echo -e "${PURPLE}[$1]${NC} $2"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get Python command (considering conda environment)
get_python_cmd() {
    local skip_conda=${1:-false}
    
    if [ "$skip_conda" = false ] && conda env list 2>/dev/null | grep -q "^${CONDA_ENV_NAME}"; then
        echo "conda run -n $CONDA_ENV_NAME python"
    else
        echo "python"
    fi
}

# Function to get pip command (considering conda environment) 
get_pip_cmd() {
    local skip_conda=${1:-false}
    
    if [ "$skip_conda" = false ] && conda env list 2>/dev/null | grep -q "^${CONDA_ENV_NAME}"; then
        echo "conda run -n $CONDA_ENV_NAME pip"
    else
        echo "pip"
    fi
}

# Function to check if conda environment exists
conda_env_exists() {
    conda env list 2>/dev/null | grep -q "^${CONDA_ENV_NAME}"
}

# Function to handle script errors
handle_error() {
    local exit_code=$?
    local line_no=$1
    print_error "Script failed at line $line_no with exit code $exit_code"
    exit $exit_code
}

# Set up error handling
trap 'handle_error ${LINENO}' ERR