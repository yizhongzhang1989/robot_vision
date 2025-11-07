#!/bin/bash

# Robot Vision Toolkit - Testing and Validation
# ==============================================
# Runs the FlowFormer++ keypoint tracker example to validate installation

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/setup_utils.sh"

# Note: We don't use 'set -e' here to provide better error reporting

# Test configurations
TEST_TIMEOUT=300  # 5 minutes for the test

# Function to run the FlowFormer++ example
run_ffpp_example() {
    local skip_conda=${1:-false}
    
    print_step "Running FlowFormer++ keypoint tracker example..."
    
    local python_cmd=$(get_python_cmd "$skip_conda")
    
    # Check if CUDA is available before running the test
    print_progress "Checking CUDA availability..."
    if $python_cmd -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        print_success "CUDA is available ✓"
        print_info "This may take a few minutes to load models and process images..."
    else
        print_warning "CUDA is not available"
        print_warning "Skipping FlowFormer++ example test (requires CUDA)"
        print_info "You can manually test later when CUDA is available by running:"
        print_info "  python examples/ffpp_keypoint_tracker_example.py"
        return 0
    fi
    
    echo ""
    cd "$PROJECT_ROOT"

    
    print_info "This may take a few minutes to load models and process images..."
    echo ""
    
    if [ ! -f "examples/ffpp_keypoint_tracker_example.py" ]; then
        print_error "Example not found: examples/ffpp_keypoint_tracker_example.py"
        return 1
    fi
    
    # Run with real-time output
    if timeout $TEST_TIMEOUT $python_cmd examples/ffpp_keypoint_tracker_example.py; then
        echo ""
        print_success "FlowFormer++ example completed successfully! ✓"
        return 0
    else
        local exit_code=$?
        echo ""
        if [ $exit_code -eq 124 ]; then
            print_error "FlowFormer++ example timed out after $TEST_TIMEOUT seconds"
        else
            print_error "FlowFormer++ example failed with exit code: $exit_code"
        fi
        return 1
    fi
}

# Main execution
main() {
    local action=${1:-all}
    local skip_conda=false
    
    # Parse arguments
    shift 2>/dev/null || true
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-conda)
                skip_conda=true
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    case "$action" in
        "all"|"quick"|"examples"|"test")
            print_banner "Robot Vision - Installation Validation"
            run_ffpp_example "$skip_conda"
            ;;
        *)
            print_banner "Robot Vision - Installation Validation"
            echo "Usage: $0 [all|quick|examples|test] [--skip-conda]"
            echo ""
            echo "This script validates the installation by running:"
            echo "  • examples/ffpp_keypoint_tracker_example.py"
            echo ""
            echo "The test will load FlowFormer++ models and process sample images."
            echo "It may take a few minutes to complete."
            run_ffpp_example "$skip_conda"
            ;;
    esac
}

# Execute if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
