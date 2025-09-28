#!/bin/bash

# Robot Vision Toolkit - Submodule Management
# ===========================================
# Updates and manages git submodules

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/setup_utils.sh"

set -e  # Exit on any error

# Function to update git submodules
update_submodules() {
    print_step "Updating git submodules..."
    
    cd "$PROJECT_ROOT"
    
    print_progress "Initializing submodules..."
    git submodule init
    
    print_progress "Updating submodules recursively..."
    git submodule update --recursive --progress
    
    # Verify submodules were updated
    verify_submodules
    
    print_success "All submodules updated successfully!"
}

# Function to verify submodules
verify_submodules() {
    print_progress "Verifying submodules..."
    
    local missing_submodules=()
    local submodule_checks=(
        "ThirdParty/camera_calibration_toolkit/README.md:camera_calibration_toolkit"
        "ThirdParty/FlowFormerPlusPlusServer/README.md:FlowFormerPlusPlusServer"
        "ThirdParty/ImageLabelingWeb/README.md:ImageLabelingWeb"
    )
    
    for check in "${submodule_checks[@]}"; do
        local file_path=$(echo "$check" | cut -d: -f1)
        local submodule_name=$(echo "$check" | cut -d: -f2)
        
        if [ ! -f "$PROJECT_ROOT/$file_path" ]; then
            missing_submodules+=("$submodule_name")
        else
            print_info "✓ $submodule_name"
        fi
    done
    
    if [ ${#missing_submodules[@]} -gt 0 ]; then
        print_error "Failed to update submodules: ${missing_submodules[*]}"
        print_info "You may need to check your network connection or submodule URLs"
        print_info "Try running: git submodule update --init --recursive --force"
        return 1
    fi
    
    print_success "All submodules verified successfully!"
}

# Function to clean submodules (reset to clean state)
clean_submodules() {
    print_step "Cleaning submodules..."
    
    cd "$PROJECT_ROOT"
    
    print_progress "Cleaning submodule directories..."
    git submodule foreach --recursive 'git clean -xfd'
    git submodule foreach --recursive 'git reset --hard'
    
    print_progress "Re-syncing submodule URLs..."
    git submodule sync --recursive
    
    print_success "Submodules cleaned successfully!"
}

# Function to show submodule status
show_submodule_status() {
    print_step "Checking submodule status..."
    
    cd "$PROJECT_ROOT"
    
    echo -e "${CYAN}Git submodule status:${NC}"
    git submodule status --recursive
    
    echo -e "\n${CYAN}Submodule summary:${NC}"
    git submodule summary
}

# Main execution
main() {
    local action=${1:-update}
    
    case "$action" in
        "update")
            print_banner "Git Submodule Update"
            update_submodules
            ;;
        "verify")
            print_banner "Submodule Verification"
            verify_submodules
            ;;
        "clean")
            print_banner "Submodule Cleanup"
            clean_submodules
            ;;
        "status")
            print_banner "Submodule Status"
            show_submodule_status
            ;;
        *)
            print_error "Unknown action: $action"
            echo "Usage: $0 [update|verify|clean|status]"
            return 1
            ;;
    esac
}

# Execute if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi