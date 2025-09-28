#!/bin/bash

# Robot Vision Toolkit - Complete Setup Script
# ============================================
# Orchestrates the entire setup process using modular scripts

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/scripts/setup_utils.sh"

set -e  # Exit on any error

# Script configuration
SETUP_LOG_FILE="$PROJECT_ROOT/setup.log"
SETUP_SCRIPTS_DIR="$PROJECT_ROOT/scripts"

# Available setup steps
declare -a SETUP_STEPS=(
    "check_requirements"
    "setup_submodules"
    "setup_conda"
    "install_dependencies"
    "download_models"
    "run_tests"
)

declare -A STEP_DESCRIPTIONS=(
    ["check_requirements"]="Verify system requirements and prerequisites"
    ["setup_submodules"]="Initialize and update Git submodules"
    ["setup_conda"]="Create and configure Conda environment"
    ["install_dependencies"]="Install Python dependencies and packages"
    ["download_models"]="Download required models and assets"
    ["run_tests"]="Run tests and validate installation"
)

declare -A STEP_SCRIPTS=(
    ["check_requirements"]="check_requirements.sh"
    ["setup_submodules"]="setup_submodules.sh"
    ["setup_conda"]="setup_conda.sh"
    ["install_dependencies"]="install_dependencies.sh"
    ["download_models"]="download_models.sh"
    ["run_tests"]="run_tests.sh"
)

# Configuration variables
SKIP_CONDA=false
SKIP_MODELS=false
SKIP_TESTS=false
FORCE_REINSTALL=false
QUICK_MODE=false
DEV_MODE=false
SELECTED_STEPS=()
CONDA_ENV_NAME="robot_vision"

# Function to show usage information
show_usage() {
    cat << EOF
Robot Vision Toolkit - Complete Setup Script

Usage: $0 [OPTIONS] [STEPS...]

OPTIONS:
    --help              Show this help message
    --skip-conda        Skip conda environment creation (use system Python)
    --skip-models       Skip model downloads (faster setup)
    --skip-tests        Skip test execution
    --force             Force reinstall/recreate everything
    --quick             Quick setup (skip optional components)
    --dev               Install development dependencies
    --env-name NAME     Conda environment name (default: robot_vision)
    --log FILE          Log file path (default: setup.log)
    --list-steps        List all available setup steps
    --dry-run          Show what would be done without executing

STEPS (if not specified, all steps will be run):
$(for step in "${SETUP_STEPS[@]}"; do
    printf "    %-18s %s\n" "$step" "${STEP_DESCRIPTIONS[$step]}"
done)

EXAMPLES:
    # Complete setup
    $0

    # Quick setup without models and tests
    $0 --skip-models --skip-tests

    # Setup with system Python (no conda)
    $0 --skip-conda

    # Only setup dependencies and run tests
    $0 install_dependencies run_tests

    # Development setup
    $0 --dev

    # Force complete reinstall
    $0 --force

    # Dry run to see what would be executed
    $0 --dry-run
EOF
}

# Function to list available steps
list_steps() {
    print_banner "Available Setup Steps"
    
    for i in "${!SETUP_STEPS[@]}"; do
        local step="${SETUP_STEPS[$i]}"
        local description="${STEP_DESCRIPTIONS[$step]}"
        local script="${STEP_SCRIPTS[$step]}"
        
        echo -e "${CYAN}$((i+1)). ${step}${NC}"
        echo -e "   Description: $description"
        echo -e "   Script: $script"
        echo
    done
}

# Function to validate step names
validate_steps() {
    local invalid_steps=()
    
    for step in "${SELECTED_STEPS[@]}"; do
        if [[ ! " ${SETUP_STEPS[*]} " =~ " $step " ]]; then
            invalid_steps+=("$step")
        fi
    done
    
    if [ ${#invalid_steps[@]} -gt 0 ]; then
        print_error "Invalid step(s): ${invalid_steps[*]}"
        echo "Available steps: ${SETUP_STEPS[*]}"
        return 1
    fi
    
    return 0
}

# Function to check if script exists and is executable
check_script() {
    local script_name=$1
    local script_path="$SETUP_SCRIPTS_DIR/$script_name"
    
    if [ ! -f "$script_path" ]; then
        print_error "Setup script not found: $script_path"
        return 1
    fi
    
    if [ ! -x "$script_path" ]; then
        print_warning "Making script executable: $script_name"
        chmod +x "$script_path"
    fi
    
    return 0
}

# Function to execute a setup step
execute_step() {
    local step_name=$1
    local script_name="${STEP_SCRIPTS[$step_name]}"
    local script_path="$SETUP_SCRIPTS_DIR/$script_name"
    
    print_step "Executing: $step_name"
    print_info "Script: $script_name"
    
    # Check if script exists
    if ! check_script "$script_name"; then
        return 1
    fi
    
    # Build command arguments based on step and configuration
    local cmd_args=()
    
    case "$step_name" in
        "check_requirements")
            # No special arguments needed
            ;;
        "setup_submodules")
            cmd_args+=("update")
            ;;
        "setup_conda")
            if [ "$FORCE_REINSTALL" = true ]; then
                cmd_args+=("create" "$CONDA_ENV_NAME" "true")
            else
                cmd_args+=("create" "$CONDA_ENV_NAME")
            fi
            ;;
        "install_dependencies")
            cmd_args+=("install" "$SKIP_CONDA" "$DEV_MODE")
            ;;
        "download_models")
            if [ "$SKIP_MODELS" = true ]; then
                print_info "Skipping model downloads (--skip-models specified)"
                return 0
            fi
            cmd_args+=("download")
            ;;
        "run_tests")
            if [ "$SKIP_TESTS" = true ]; then
                print_info "Skipping tests (--skip-tests specified)"
                return 0
            fi
            if [ "$QUICK_MODE" = true ]; then
                cmd_args+=("quick" "$SKIP_CONDA")
            else
                cmd_args+=("all" "$SKIP_CONDA" "false")
            fi
            ;;
    esac
    
    # Execute the step
    local start_time=$(date +%s)
    
    if [ "$DRY_RUN" = true ]; then
        print_info "DRY RUN: Would execute: $script_path ${cmd_args[*]}"
        return 0
    fi
    
    print_progress "Running: $script_path ${cmd_args[*]}"
    
    if "$script_path" "${cmd_args[@]}" 2>&1 | tee -a "$SETUP_LOG_FILE"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "Step completed: $step_name (${duration}s)"
        return 0
    else
        local exit_code=$?
        print_error "Step failed: $step_name (exit code: $exit_code)"
        return $exit_code
    fi
}

# Function to run complete setup
run_complete_setup() {
    local steps_to_run=("${SELECTED_STEPS[@]}")
    
    # If no specific steps selected, run all steps
    if [ ${#steps_to_run[@]} -eq 0 ]; then
        steps_to_run=("${SETUP_STEPS[@]}")
    fi
    
    print_banner "Robot Vision Toolkit Setup"
    
    # Show configuration
    print_step "Setup Configuration"
    echo -e "  Project Root: $PROJECT_ROOT"
    echo -e "  Skip Conda: $SKIP_CONDA"
    echo -e "  Skip Models: $SKIP_MODELS"
    echo -e "  Skip Tests: $SKIP_TESTS"
    echo -e "  Force Reinstall: $FORCE_REINSTALL"
    echo -e "  Quick Mode: $QUICK_MODE"
    echo -e "  Development Mode: $DEV_MODE"
    echo -e "  Conda Environment: $CONDA_ENV_NAME"
    echo -e "  Log File: $SETUP_LOG_FILE"
    echo -e "  Steps to run: ${steps_to_run[*]}"
    echo
    
    if [ "$DRY_RUN" = true ]; then
        print_warning "DRY RUN MODE - No actual changes will be made"
        echo
    fi
    
    # Initialize log file
    if [ "$DRY_RUN" = false ]; then
        echo "Robot Vision Toolkit Setup Log - $(date)" > "$SETUP_LOG_FILE"
        echo "Configuration: skip_conda=$SKIP_CONDA, skip_models=$SKIP_MODELS, skip_tests=$SKIP_TESTS" >> "$SETUP_LOG_FILE"
        echo "========================================" >> "$SETUP_LOG_FILE"
    fi
    
    # Execute steps
    local total_steps=${#steps_to_run[@]}
    local completed_steps=0
    local failed_steps=()
    local start_time=$(date +%s)
    
    for i in "${!steps_to_run[@]}"; do
        local step="${steps_to_run[$i]}"
        local step_num=$((i + 1))
        
        print_step_number "$step_num/$total_steps" "$(echo "$step" | tr '_' ' ' | tr '[:lower:]' '[:upper:]')"
        
        if execute_step "$step"; then
            ((completed_steps++))
        else
            failed_steps+=("$step")
            
            # Ask user if they want to continue on failure
            if [ ${#steps_to_run[@]} -gt 1 ] && [ $((i + 1)) -lt ${#steps_to_run[@]} ]; then
                echo
                print_warning "Step failed: $step"
                if [ "$DRY_RUN" = false ]; then
                    echo -n "Continue with remaining steps? (y/N): "
                    read -r continue_choice
                    if [[ ! "$continue_choice" =~ ^[Yy]$ ]]; then
                        print_info "Setup aborted by user"
                        break
                    fi
                fi
            fi
        fi
        echo
    done
    
    # Show summary
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    print_step "Setup Summary"
    echo -e "  Total steps: $total_steps"
    echo -e "  Completed: $completed_steps"
    echo -e "  Failed: ${#failed_steps[@]}"
    echo -e "  Duration: ${total_duration}s"
    
    if [ ${#failed_steps[@]} -gt 0 ]; then
        echo -e "  Failed steps: ${failed_steps[*]}"
    fi
    
    echo -e "  Log file: $SETUP_LOG_FILE"
    echo
    
    if [ ${#failed_steps[@]} -eq 0 ]; then
        print_success "Setup completed successfully! 🎉"
        
        if [ "$SKIP_TESTS" = false ] && [ "$DRY_RUN" = false ]; then
            echo
            print_info "You can now:"
            echo "  • Run examples: cd examples && python keypoint_tracker_simple.py"
            echo "  • Start web services: python scripts/video_task.py"
            echo "  • Test installation: scripts/run_tests.sh quick"
        fi
        
        return 0
    else
        print_error "Setup completed with errors!"
        print_info "Check the log file for details: $SETUP_LOG_FILE"
        return 1
    fi
}

# Parse command line arguments
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            exit 0
            ;;
        --skip-conda)
            SKIP_CONDA=true
            shift
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --force)
            FORCE_REINSTALL=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            SKIP_MODELS=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --env-name)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        --log)
            SETUP_LOG_FILE="$2"
            shift 2
            ;;
        --list-steps)
            list_steps
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            SELECTED_STEPS+=("$1")
            shift
            ;;
    esac
done

# Validate selected steps
if [ ${#SELECTED_STEPS[@]} -gt 0 ]; then
    if ! validate_steps; then
        exit 1
    fi
fi

# Update environment variables
export CONDA_ENV_NAME
export SKIP_CONDA

# Run setup
run_complete_setup