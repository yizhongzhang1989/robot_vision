#!/bin/bash

# Robot Vision Toolkit - Testing and Validation
# ==============================================
# Runs tests and validates the installation

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/setup_utils.sh"

set -e  # Exit on any error

# Test configurations
SAMPLE_IMAGE_URL="https://github.com/pytorch/vision/raw/main/gallery/assets/dog1.jpg"
TEST_TIMEOUT=300  # 5 minutes for each test

# Function to run basic Python import tests
run_import_tests() {
    local skip_conda=${1:-false}
    
    print_step "Running Python import tests..."
    
    local python_cmd=$(get_python_cmd "$skip_conda")
    
    # List of critical imports to test
    local imports=(
        "numpy"
        "cv2"
        "PIL"
        "matplotlib"
        "scipy"
        "torch"
        "torchvision"
        "flask"
        "requests"
    )
    
    local passed=0
    local failed=0
    
    for import_name in "${imports[@]}"; do
        print_progress "Testing import: $import_name"
        
        if timeout 30 $python_cmd -c "import $import_name; print(f'✓ {import_name} version: {getattr($import_name, \"__version__\", \"unknown\")}')" 2>/dev/null; then
            ((passed++))
        else
            print_warning "Failed to import: $import_name"
            ((failed++))
        fi
    done
    
    echo
    print_info "Import test results: $passed passed, $failed failed"
    
    if [ $failed -eq 0 ]; then
        print_success "All critical imports working!"
        return 0
    else
        print_warning "Some imports failed - this may affect functionality"
        return 1
    fi
}

# Function to test core robot vision functionality
test_core_functionality() {
    local skip_conda=${1:-false}
    
    print_step "Testing core robot vision functionality..."
    
    local python_cmd=$(get_python_cmd "$skip_conda")
    cd "$PROJECT_ROOT"
    
    # Test keypoint tracker
    print_progress "Testing keypoint tracker..."
    if $python_cmd -c "
import sys
sys.path.insert(0, '.')
from core.keypoint_tracker import KeypointTracker
tracker = KeypointTracker()
print('✓ KeypointTracker initialized successfully')
" 2>/dev/null; then
        print_success "Core keypoint tracker test passed"
    else
        print_error "Core keypoint tracker test failed"
        return 1
    fi
    
    # Test utilities
    print_progress "Testing utilities..."
    if $python_cmd -c "
import sys
sys.path.insert(0, '.')
from core.utils import validate_image_path, create_output_dir
print('✓ Core utilities imported successfully')
" 2>/dev/null; then
        print_success "Core utilities test passed"
    else
        print_error "Core utilities test failed"
        return 1
    fi
    
    return 0
}

# Function to test FlowFormer++ server
test_flowformer_server() {
    local skip_conda=${1:-false}
    
    print_step "Testing FlowFormer++ server..."
    
    local python_cmd=$(get_python_cmd "$skip_conda")
    local ffpp_dir="$PROJECT_ROOT/ThirdParty/FlowFormerPlusPlusServer"
    
    if [ ! -d "$ffpp_dir" ]; then
        print_warning "FlowFormerPlusPlusServer not found, skipping test"
        return 0
    fi
    
    cd "$ffpp_dir"
    
    # Test FlowFormer API import
    print_progress "Testing FlowFormer API import..."
    if timeout 60 $python_cmd -c "
import sys
import os
sys.path.insert(0, '.')
try:
    from flowformer_api import FlowFormerAPI
    print('✓ FlowFormerAPI imported successfully')
except Exception as e:
    print(f'✗ FlowFormerAPI import failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        print_success "FlowFormer API test passed"
    else
        print_warning "FlowFormer API test failed - check model availability"
        return 1
    fi
    
    return 0
}

# Function to test camera calibration toolkit
test_calibration_toolkit() {
    local skip_conda=${1:-false}
    
    print_step "Testing camera calibration toolkit..."
    
    local python_cmd=$(get_python_cmd "$skip_conda")
    local calib_dir="$PROJECT_ROOT/ThirdParty/camera_calibration_toolkit"
    
    if [ ! -d "$calib_dir" ]; then
        print_warning "Camera calibration toolkit not found, skipping test"
        return 0
    fi
    
    cd "$calib_dir"
    
    # Test calibration imports
    print_progress "Testing calibration imports..."
    if timeout 30 $python_cmd -c "
import sys
sys.path.insert(0, '.')
try:
    from core.calibration_factory import CalibrationFactory
    print('✓ CalibrationFactory imported successfully')
except Exception as e:
    print(f'✗ CalibrationFactory import failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        print_success "Camera calibration test passed"
    else
        print_warning "Camera calibration test failed"
        return 1
    fi
    
    return 0
}

# Function to test web services
test_web_services() {
    local skip_conda=${1:-false}
    local quick_test=${2:-true}
    
    print_step "Testing web services..."
    
    local python_cmd=$(get_python_cmd "$skip_conda")
    
    # Test FlowFormer web service
    local ffpp_dir="$PROJECT_ROOT/ThirdParty/FlowFormerPlusPlusServer"
    if [ -d "$ffpp_dir" ]; then
        print_progress "Testing FlowFormer web service..."
        cd "$ffpp_dir"
        
        if $python_cmd -c "
import sys
sys.path.insert(0, '.')
from app import app
print('✓ FlowFormer Flask app imported successfully')
" 2>/dev/null; then
            print_success "FlowFormer web service test passed"
        else
            print_warning "FlowFormer web service test failed"
        fi
    fi
    
    # Test Image Labeling Web service
    local labeling_dir="$PROJECT_ROOT/ThirdParty/ImageLabelingWeb"
    if [ -d "$labeling_dir" ]; then
        print_progress "Testing Image Labeling web service..."
        cd "$labeling_dir"
        
        if $python_cmd -c "
import sys
sys.path.insert(0, '.')
from launch_server import create_app
print('✓ Image Labeling Flask app imported successfully')
" 2>/dev/null; then
            print_success "Image Labeling web service test passed"
        else
            print_warning "Image Labeling web service test failed"
        fi
    fi
    
    # Test camera calibration web service
    local calib_dir="$PROJECT_ROOT/ThirdParty/camera_calibration_toolkit"
    if [ -d "$calib_dir/web" ]; then
        print_progress "Testing camera calibration web service..."
        cd "$calib_dir"
        
        if [ -f "web/app.py" ] && $python_cmd -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'web')
from web.app import app
print('✓ Camera calibration Flask app imported successfully')
" 2>/dev/null; then
            print_success "Camera calibration web service test passed"
        else
            print_info "Camera calibration web service not available or test failed"
        fi
    fi
    
    return 0
}

# Function to test examples
test_examples() {
    local skip_conda=${1:-false}
    
    print_step "Testing example scripts..."
    
    local python_cmd=$(get_python_cmd "$skip_conda")
    cd "$PROJECT_ROOT"
    
    # Test example imports (without actually running them)
    local examples=(
        "examples/keypoint_tracker_simple.py"
        "examples/keypoint_tracker_with_validation.py"
    )
    
    local passed=0
    local failed=0
    
    for example in "${examples[@]}"; do
        if [ -f "$example" ]; then
            print_progress "Testing example: $(basename "$example")"
            
            # Test if the example can be imported without errors
            if timeout 30 $python_cmd -c "
import sys
sys.path.insert(0, '.')
import ast
import importlib.util

# Parse the file to check for syntax errors
with open('$example', 'r') as f:
    try:
        ast.parse(f.read())
        print('✓ $(basename "$example") - syntax check passed')
    except SyntaxError as e:
        print(f'✗ $(basename "$example") - syntax error: {e}')
        sys.exit(1)
" 2>/dev/null; then
                ((passed++))
            else
                print_warning "Example test failed: $(basename "$example")"
                ((failed++))
            fi
        else
            print_warning "Example not found: $example"
            ((failed++))
        fi
    done
    
    echo
    print_info "Example test results: $passed passed, $failed failed"
    return $failed
}

# Function to run comprehensive system test
run_system_test() {
    local skip_conda=${1:-false}
    
    print_step "Running comprehensive system test..."
    
    local python_cmd=$(get_python_cmd "$skip_conda")
    cd "$PROJECT_ROOT"
    
    # Create a temporary test directory
    local test_dir="$PROJECT_ROOT/temp_test_$$"
    mkdir -p "$test_dir"
    
    # Cleanup function
    cleanup_test() {
        rm -rf "$test_dir"
    }
    trap cleanup_test EXIT
    
    print_progress "Creating test data..."
    
    # Create a simple test image if sample data doesn't exist
    if [ ! -f "sample_data/flow_image_pair/ref_img.jpg" ]; then
        print_progress "Generating test image..."
        $python_cmd -c "
import numpy as np
from PIL import Image
import os

# Create a simple test image
img = np.zeros((300, 400, 3), dtype=np.uint8)
img[50:250, 50:350] = [100, 150, 200]  # Blue rectangle
img[100:200, 100:300] = [200, 100, 50]  # Orange rectangle

os.makedirs('$test_dir', exist_ok=True)
Image.fromarray(img).save('$test_dir/test_image.jpg')
print('✓ Test image created')
"
    else
        cp "sample_data/flow_image_pair/ref_img.jpg" "$test_dir/test_image.jpg"
        print_success "Using existing sample image"
    fi
    
    # Test basic keypoint detection
    print_progress "Testing keypoint detection..."
    if $python_cmd -c "
import sys
sys.path.insert(0, '.')
import cv2
import numpy as np
from core.keypoint_tracker import KeypointTracker

# Load test image
img = cv2.imread('$test_dir/test_image.jpg')
if img is None:
    raise Exception('Failed to load test image')

# Initialize tracker
tracker = KeypointTracker()

# Detect keypoints
keypoints = tracker.detect_keypoints(img)
print(f'✓ Detected {len(keypoints)} keypoints')

if len(keypoints) >= 10:
    print('✓ Sufficient keypoints detected for tracking')
else:
    print('⚠ Few keypoints detected - may affect tracking quality')
" 2>/dev/null; then
        print_success "Keypoint detection test passed"
    else
        print_error "Keypoint detection test failed"
        return 1
    fi
    
    print_success "System test completed successfully!"
    return 0
}

# Function to run all tests
run_all_tests() {
    local skip_conda=${1:-false}
    local quick_mode=${2:-false}
    
    print_banner "Running All Tests"
    
    local total_tests=0
    local passed_tests=0
    local test_results=()
    
    # Import tests
    ((total_tests++))
    print_step_number $total_tests "Import Tests"
    if run_import_tests "$skip_conda"; then
        ((passed_tests++))
        test_results+=("✓ Import tests")
    else
        test_results+=("✗ Import tests")
    fi
    
    # Core functionality tests
    ((total_tests++))
    print_step_number $total_tests "Core Functionality"
    if test_core_functionality "$skip_conda"; then
        ((passed_tests++))
        test_results+=("✓ Core functionality")
    else
        test_results+=("✗ Core functionality")
    fi
    
    # Example tests
    ((total_tests++))
    print_step_number $total_tests "Examples"
    if test_examples "$skip_conda"; then
        ((passed_tests++))
        test_results+=("✓ Examples")
    else
        test_results+=("✗ Examples")
    fi
    
    # Web services tests (skip in quick mode)
    if [ "$quick_mode" = false ]; then
        ((total_tests++))
        print_step_number $total_tests "Web Services"
        if test_web_services "$skip_conda" true; then
            ((passed_tests++))
            test_results+=("✓ Web services")
        else
            test_results+=("✗ Web services")
        fi
        
        # FlowFormer tests
        ((total_tests++))
        print_step_number $total_tests "FlowFormer++"
        if test_flowformer_server "$skip_conda"; then
            ((passed_tests++))
            test_results+=("✓ FlowFormer++")
        else
            test_results+=("✗ FlowFormer++")
        fi
        
        # Calibration tests
        ((total_tests++))
        print_step_number $total_tests "Camera Calibration"
        if test_calibration_toolkit "$skip_conda"; then
            ((passed_tests++))
            test_results+=("✓ Camera calibration")
        else
            test_results+=("✗ Camera calibration")
        fi
        
        # System tests
        ((total_tests++))
        print_step_number $total_tests "System Integration"
        if run_system_test "$skip_conda"; then
            ((passed_tests++))
            test_results+=("✓ System integration")
        else
            test_results+=("✗ System integration")
        fi
    fi
    
    # Test summary
    print_step "Test Summary"
    echo
    for result in "${test_results[@]}"; do
        echo "  $result"
    done
    echo
    print_info "Tests passed: $passed_tests/$total_tests"
    
    if [ $passed_tests -eq $total_tests ]; then
        print_success "All tests passed! 🎉"
        return 0
    else
        print_warning "Some tests failed. See details above."
        return 1
    fi
}

# Function to run quick validation
run_quick_validation() {
    local skip_conda=${1:-false}
    
    print_banner "Quick Validation"
    
    # Just check critical imports and core functionality
    if run_import_tests "$skip_conda" && test_core_functionality "$skip_conda"; then
        print_success "Quick validation passed! ✓"
        return 0
    else
        print_error "Quick validation failed! ✗"
        return 1
    fi
}

# Function to benchmark performance
run_performance_test() {
    local skip_conda=${1:-false}
    
    print_step "Running performance benchmark..."
    
    local python_cmd=$(get_python_cmd "$skip_conda")
    cd "$PROJECT_ROOT"
    
    # Create benchmark test
    $python_cmd -c "
import time
import sys
sys.path.insert(0, '.')

print('Performance Benchmark Results:')
print('=' * 40)

# Test import speed
start_time = time.time()
import numpy as np
import cv2
from core.keypoint_tracker import KeypointTracker
import_time = time.time() - start_time
print(f'Import time: {import_time:.3f} seconds')

# Test initialization speed
start_time = time.time()
tracker = KeypointTracker()
init_time = time.time() - start_time
print(f'Tracker init time: {init_time:.3f} seconds')

# Test keypoint detection speed
test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
start_time = time.time()
keypoints = tracker.detect_keypoints(test_img)
detection_time = time.time() - start_time
print(f'Keypoint detection time: {detection_time:.3f} seconds')
print(f'Detected keypoints: {len(keypoints)}')

print('=' * 40)
total_time = import_time + init_time + detection_time
print(f'Total benchmark time: {total_time:.3f} seconds')

if total_time < 5.0:
    print('✓ Performance: GOOD')
elif total_time < 10.0:
    print('⚠ Performance: ACCEPTABLE')
else:
    print('✗ Performance: SLOW')
"
}

# Main execution
main() {
    local action=${1:-all}
    local skip_conda=${2:-false}
    local quick_mode=${3:-false}
    
    case "$action" in
        "all")
            run_all_tests "$skip_conda" "$quick_mode"
            ;;
        "quick")
            run_quick_validation "$skip_conda"
            ;;
        "imports")
            print_banner "Import Tests"
            run_import_tests "$skip_conda"
            ;;
        "core")
            print_banner "Core Functionality Tests"
            test_core_functionality "$skip_conda"
            ;;
        "flowformer")
            print_banner "FlowFormer++ Tests"
            test_flowformer_server "$skip_conda"
            ;;
        "calibration")
            print_banner "Camera Calibration Tests"
            test_calibration_toolkit "$skip_conda"
            ;;
        "web")
            print_banner "Web Services Tests"
            test_web_services "$skip_conda" false
            ;;
        "examples")
            print_banner "Example Tests"
            test_examples "$skip_conda"
            ;;
        "system")
            print_banner "System Integration Tests"
            run_system_test "$skip_conda"
            ;;
        "performance")
            print_banner "Performance Benchmark"
            run_performance_test "$skip_conda"
            ;;
        *)
            print_error "Unknown test action: $action"
            echo "Usage: $0 [all|quick|imports|core|flowformer|calibration|web|examples|system|performance] [skip_conda] [quick_mode]"
            echo ""
            echo "Test Actions:"
            echo "  all         - Run all tests (default)"
            echo "  quick       - Quick validation (imports + core)"
            echo "  imports     - Test Python imports"
            echo "  core        - Test core functionality"
            echo "  flowformer  - Test FlowFormer++ server"
            echo "  calibration - Test camera calibration"
            echo "  web         - Test web services"
            echo "  examples    - Test example scripts"
            echo "  system      - Test system integration"
            echo "  performance - Run performance benchmark"
            return 1
            ;;
    esac
}

# Execute if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi