#!/bin/bash

# Robot Vision Toolkit - Model and Asset Downloads
# =================================================
# Downloads required models and assets for all components

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/setup_utils.sh"

set -e  # Exit on any error

# Model configurations
declare -A FLOWFORMER_MODELS=(
    ["flowformer_sintel.pth"]="https://github.com/drinkingcoder/FlowFormerPlusPlus/releases/download/v1.0/flowformer_sintel.pth"
    ["flowformer_things.pth"]="https://github.com/drinkingcoder/FlowFormerPlusPlus/releases/download/v1.0/flowformer_things.pth"
    ["flowformer_kitti.pth"]="https://github.com/drinkingcoder/FlowFormerPlusPlus/releases/download/v1.0/flowformer_kitti.pth"
)

declare -A MODEL_SIZES=(
    ["flowformer_sintel.pth"]="350MB"
    ["flowformer_things.pth"]="350MB"
    ["flowformer_kitti.pth"]="350MB"
)

declare -A MODEL_DESCRIPTIONS=(
    ["flowformer_sintel.pth"]="FlowFormer++ trained on Sintel dataset (best for synthetic/clean images)"
    ["flowformer_things.pth"]="FlowFormer++ trained on FlyingThings3D (general purpose)"
    ["flowformer_kitti.pth"]="FlowFormer++ trained on KITTI dataset (best for automotive/outdoor)"
)

# Function to check available disk space
check_disk_space() {
    local required_mb=${1:-2000}  # Default 2GB
    local target_dir=${2:-"$PROJECT_ROOT/ThirdParty/FlowFormerPlusPlusServer/checkpoints"}
    
    print_progress "Checking available disk space..."
    
    # Create target directory if it doesn't exist
    mkdir -p "$target_dir"
    
    local available_kb=$(df "$target_dir" | awk 'NR==2 {print $4}')
    local available_mb=$((available_kb / 1024))
    
    if [ "$available_mb" -lt "$required_mb" ]; then
        print_error "Insufficient disk space!"
        print_error "Required: ${required_mb}MB, Available: ${available_mb}MB"
        return 1
    else
        print_success "Sufficient disk space available: ${available_mb}MB"
        return 0
    fi
}

# Function to download a file with progress and resume capability
download_file() {
    local url=$1
    local output_path=$2
    local description=${3:-"file"}
    
    if [ -z "$url" ] || [ -z "$output_path" ]; then
        print_error "URL and output path required"
        return 1
    fi
    
    print_progress "Downloading $description..."
    print_info "URL: $url"
    print_info "Destination: $output_path"
    
    # Create output directory if it doesn't exist
    mkdir -p "$(dirname "$output_path")"
    
    # Check if file already exists and get size
    local resume_option=""
    if [ -f "$output_path" ]; then
        local existing_size=$(stat -c%s "$output_path" 2>/dev/null || echo 0)
        if [ "$existing_size" -gt 0 ]; then
            print_info "Existing file found (${existing_size} bytes), attempting to resume..."
            resume_option="-C -"
        fi
    fi
    
    # Download with curl (preferred) or wget
    if command_exists curl; then
        if ! curl -L $resume_option --progress-bar --fail "$url" -o "$output_path"; then
            print_error "Download failed with curl"
            return 1
        fi
    elif command_exists wget; then
        local wget_resume=""
        [ -n "$resume_option" ] && wget_resume="-c"
        if ! wget $wget_resume --progress=bar:force:noscroll "$url" -O "$output_path"; then
            print_error "Download failed with wget"
            return 1
        fi
    else
        print_error "Neither curl nor wget is available for downloading"
        return 1
    fi
    
    print_success "Downloaded: $description"
    return 0
}

# Function to verify file integrity (basic size check)
verify_file() {
    local file_path=$1
    local min_size=${2:-1000000}  # Default 1MB minimum
    
    if [ ! -f "$file_path" ]; then
        print_error "File does not exist: $file_path"
        return 1
    fi
    
    local file_size=$(stat -c%s "$file_path" 2>/dev/null || echo 0)
    if [ "$file_size" -lt "$min_size" ]; then
        print_error "File appears to be corrupted or incomplete: $file_path"
        print_error "Size: ${file_size} bytes (expected > ${min_size})"
        return 1
    fi
    
    print_success "File verification passed: $(basename "$file_path") (${file_size} bytes)"
    return 0
}

# Function to download FlowFormer++ models
download_flowformer_models() {
    local model_name=${1:-"all"}
    local force_download=${2:-false}
    
    print_step "Downloading FlowFormer++ models..."
    
    local checkpoint_dir="$PROJECT_ROOT/ThirdParty/FlowFormerPlusPlusServer/checkpoints"
    
    # Check if FlowFormerPlusPlusServer exists
    if [ ! -d "$PROJECT_ROOT/ThirdParty/FlowFormerPlusPlusServer" ]; then
        print_error "FlowFormerPlusPlusServer directory not found!"
        print_error "Please run submodule setup first: scripts/setup_submodules.sh update"
        return 1
    fi
    
    # Create checkpoints directory
    mkdir -p "$checkpoint_dir"
    
    # Check disk space (need ~1GB for all models)
    if ! check_disk_space 1200 "$checkpoint_dir"; then
        return 1
    fi
    
    local downloaded_count=0
    local skipped_count=0
    local failed_count=0
    
    # Download specific model or all models
    if [ "$model_name" == "all" ]; then
        print_info "Downloading all FlowFormer++ models..."
        for model in "${!FLOWFORMER_MODELS[@]}"; do
            download_single_model "$model" "$checkpoint_dir" "$force_download"
            case $? in
                0) ((downloaded_count++)) ;;
                2) ((skipped_count++)) ;;
                *) ((failed_count++)) ;;
            esac
        done
    else
        # Check if specific model exists in configuration
        if [ -z "${FLOWFORMER_MODELS[$model_name]}" ]; then
            print_error "Unknown model: $model_name"
            print_info "Available models:"
            for model in "${!FLOWFORMER_MODELS[@]}"; do
                echo "  - $model: ${MODEL_DESCRIPTIONS[$model]}"
            done
            return 1
        fi
        
        download_single_model "$model_name" "$checkpoint_dir" "$force_download"
        case $? in
            0) ((downloaded_count++)) ;;
            2) ((skipped_count++)) ;;
            *) ((failed_count++)) ;;
        esac
    fi
    
    # Summary
    print_step "Download Summary:"
    print_info "Downloaded: $downloaded_count"
    print_info "Skipped (already exists): $skipped_count"
    [ "$failed_count" -gt 0 ] && print_warning "Failed: $failed_count"
    
    if [ "$failed_count" -eq 0 ]; then
        print_success "Model download completed successfully!"
    else
        print_warning "Some models failed to download. Check the errors above."
        return 1
    fi
}

# Helper function to download a single model
download_single_model() {
    local model_name=$1
    local checkpoint_dir=$2
    local force_download=${3:-false}
    
    local model_url="${FLOWFORMER_MODELS[$model_name]}"
    local model_path="$checkpoint_dir/$model_name"
    local model_description="${MODEL_DESCRIPTIONS[$model_name]}"
    local model_size="${MODEL_SIZES[$model_name]}"
    
    print_progress "Processing model: $model_name ($model_size)"
    print_info "Description: $model_description"
    
    # Check if model already exists
    if [ -f "$model_path" ] && [ "$force_download" = false ]; then
        if verify_file "$model_path" 100000000; then  # 100MB minimum for model files
            print_success "Model already exists and verified: $model_name"
            return 2  # Skipped
        else
            print_warning "Existing model file appears corrupted, re-downloading..."
            rm -f "$model_path"
        fi
    elif [ -f "$model_path" ] && [ "$force_download" = true ]; then
        print_info "Force download requested, removing existing file..."
        rm -f "$model_path"
    fi
    
    # Download the model
    if download_file "$model_url" "$model_path" "$model_name"; then
        if verify_file "$model_path" 100000000; then
            return 0  # Success
        else
            rm -f "$model_path"  # Remove corrupted download
            return 1  # Failed
        fi
    else
        return 1  # Failed
    fi
}

# Function to list available models
list_models() {
    print_step "Available FlowFormer++ Models:"
    
    for model in "${!FLOWFORMER_MODELS[@]}"; do
        local status="Not downloaded"
        local model_path="$PROJECT_ROOT/ThirdParty/FlowFormerPlusPlusServer/checkpoints/$model"
        
        if [ -f "$model_path" ]; then
            if verify_file "$model_path" 100000000 >/dev/null 2>&1; then
                status="${GREEN}Downloaded and verified${NC}"
            else
                status="${YELLOW}Downloaded but may be corrupted${NC}"
            fi
        else
            status="${RED}Not downloaded${NC}"
        fi
        
        echo -e "  ${CYAN}$model${NC} (${MODEL_SIZES[$model]})"
        echo -e "    Description: ${MODEL_DESCRIPTIONS[$model]}"
        echo -e "    Status: $status"
        echo
    done
}

# Function to clean up downloaded models
clean_models() {
    local confirm=${1:-false}
    
    print_step "Cleaning up downloaded models..."
    
    local checkpoint_dir="$PROJECT_ROOT/ThirdParty/FlowFormerPlusPlusServer/checkpoints"
    
    if [ ! -d "$checkpoint_dir" ]; then
        print_info "No checkpoint directory found, nothing to clean"
        return 0
    fi
    
    # Find model files
    local model_files=()
    for model in "${!FLOWFORMER_MODELS[@]}"; do
        if [ -f "$checkpoint_dir/$model" ]; then
            model_files+=("$checkpoint_dir/$model")
        fi
    done
    
    if [ ${#model_files[@]} -eq 0 ]; then
        print_info "No model files found to clean"
        return 0
    fi
    
    # Show files to be deleted
    print_info "Files to be deleted:"
    local total_size=0
    for file in "${model_files[@]}"; do
        local size=$(stat -c%s "$file" 2>/dev/null || echo 0)
        local size_mb=$((size / 1024 / 1024))
        total_size=$((total_size + size_mb))
        echo "  - $(basename "$file") (${size_mb}MB)"
    done
    echo "  Total size: ${total_size}MB"
    
    # Confirm deletion
    if [ "$confirm" = false ]; then
        echo -n "Are you sure you want to delete these files? (y/N): "
        read -r confirmation
        if [[ ! "$confirmation" =~ ^[Yy]$ ]]; then
            print_info "Cleanup cancelled"
            return 0
        fi
    fi
    
    # Delete files
    for file in "${model_files[@]}"; do
        rm -f "$file"
        print_info "Deleted: $(basename "$file")"
    done
    
    print_success "Model cleanup completed! Freed ${total_size}MB of disk space."
}

# Function to check model status
check_model_status() {
    print_step "Model Status Check:"
    
    local checkpoint_dir="$PROJECT_ROOT/ThirdParty/FlowFormerPlusPlusServer/checkpoints"
    local total_models=${#FLOWFORMER_MODELS[@]}
    local downloaded_models=0
    local verified_models=0
    
    for model in "${!FLOWFORMER_MODELS[@]}"; do
        local model_path="$checkpoint_dir/$model"
        
        if [ -f "$model_path" ]; then
            ((downloaded_models++))
            if verify_file "$model_path" 100000000 >/dev/null 2>&1; then
                ((verified_models++))
            fi
        fi
    done
    
    echo -e "  Total models: $total_models"
    echo -e "  Downloaded: $downloaded_models"
    echo -e "  Verified: $verified_models"
    
    if [ "$verified_models" -eq "$total_models" ]; then
        print_success "All models are downloaded and verified!"
        return 0
    elif [ "$downloaded_models" -gt 0 ]; then
        print_warning "Some models are missing or corrupted"
        return 1
    else
        print_info "No models downloaded yet"
        return 2
    fi
}

# Main execution
main() {
    local action=${1:-download}
    local model_name=${2:-all}
    local force=${3:-false}
    
    case "$action" in
        "download")
            print_banner "Model Download"
            download_flowformer_models "$model_name" "$force"
            ;;
        "list")
            print_banner "Available Models"
            list_models
            ;;
        "status")
            print_banner "Model Status"
            check_model_status
            ;;
        "clean")
            print_banner "Model Cleanup"
            clean_models "$force"
            ;;
        "verify")
            print_banner "Model Verification"
            local checkpoint_dir="$PROJECT_ROOT/ThirdParty/FlowFormerPlusPlusServer/checkpoints"
            local exit_code=0
            for model in "${!FLOWFORMER_MODELS[@]}"; do
                local model_path="$checkpoint_dir/$model"
                if [ -f "$model_path" ]; then
                    if ! verify_file "$model_path" 100000000; then
                        exit_code=1
                    fi
                else
                    print_warning "Model not found: $model"
                    exit_code=1
                fi
            done
            exit $exit_code
            ;;
        *)
            print_error "Unknown action: $action"
            echo "Usage: $0 [download|list|status|clean|verify] [model_name] [force]"
            echo ""
            echo "Actions:"
            echo "  download [model_name] [force] - Download models (default: all models)"
            echo "  list                          - List available models and their status"
            echo "  status                        - Show download status summary"
            echo "  clean [force]                 - Remove downloaded models"
            echo "  verify                        - Verify integrity of downloaded models"
            return 1
            ;;
    esac
}

# Execute if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi