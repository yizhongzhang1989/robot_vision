#!/bin/bash

# Robot Vision Toolkit - Conda Installation Helper
# =================================================
# Automatically installs Miniconda for the project

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

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

print_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Check if conda is already installed
if command -v conda >/dev/null 2>&1; then
    print_success "Conda is already installed!"
    conda --version
    echo ""
    print_info "You can now run the setup script:"
    echo "  bash setup_all_in_one.sh"
    exit 0
fi

echo "========================================"
echo "  Miniconda Installation Script"
echo "========================================"
echo ""

# Determine architecture
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
elif [ "$ARCH" = "aarch64" ]; then
    INSTALLER="Miniconda3-latest-Linux-aarch64.sh"
else
    print_error "Unsupported architecture: $ARCH"
    print_info "Please install Conda manually from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_info "Detected architecture: $ARCH"
print_info "Installer: $INSTALLER"
echo ""

# Set installation directory
INSTALL_DIR="$HOME/miniconda3"
DOWNLOAD_DIR="/tmp"

# Parse command line arguments
BATCH_MODE=false
CUSTOM_DIR=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes)
            BATCH_MODE=true
            shift
            ;;
        -p|--prefix)
            INSTALL_DIR="$2"
            CUSTOM_DIR=true
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -y, --yes           Skip confirmation prompt (batch mode)"
            echo "  -p, --prefix PATH   Custom installation directory (default: \$HOME/miniconda3)"
            echo "  -h, --help          Show this help message"
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

# Show installation plan
print_step "Installation Plan:"
echo "  • Download: https://repo.anaconda.com/miniconda/$INSTALLER"
echo "  • Install to: $INSTALL_DIR"
echo "  • Disk space needed: ~400 MB"
echo ""

# Confirm installation
if [ "$BATCH_MODE" = false ]; then
    read -p "Continue with installation? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installation cancelled"
        exit 0
    fi
fi

# Check if installation directory already exists
if [ -d "$INSTALL_DIR" ]; then
    print_warning "Installation directory already exists: $INSTALL_DIR"
    if [ "$BATCH_MODE" = false ]; then
        read -p "Remove existing directory and continue? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Installation cancelled"
            exit 0
        fi
    fi
    print_info "Removing existing directory..."
    rm -rf "$INSTALL_DIR"
fi

# Download installer
print_step "Downloading Miniconda installer..."
cd "$DOWNLOAD_DIR" || exit 1

INSTALLER_URL="https://repo.anaconda.com/miniconda/$INSTALLER"
if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress "$INSTALLER_URL" || {
        print_error "Failed to download installer"
        exit 1
    }
elif command -v curl >/dev/null 2>&1; then
    curl -L -o "$INSTALLER" --progress-bar "$INSTALLER_URL" || {
        print_error "Failed to download installer"
        exit 1
    }
else
    print_error "Neither wget nor curl is available"
    print_info "Please install wget or curl and try again"
    exit 1
fi

print_success "Download completed"

# Install Miniconda
print_step "Installing Miniconda..."
bash "$INSTALLER" -b -p "$INSTALL_DIR" || {
    print_error "Installation failed"
    rm -f "$INSTALLER"
    exit 1
}

print_success "Miniconda installed successfully!"

# Initialize conda
print_step "Initializing Conda..."

# Initialize for bash
if [ -f "$HOME/.bashrc" ]; then
    print_info "Initializing for bash..."
    "$INSTALL_DIR/bin/conda" init bash >/dev/null 2>&1
fi

# Initialize for zsh if it exists
if [ -f "$HOME/.zshrc" ]; then
    print_info "Initializing for zsh..."
    "$INSTALL_DIR/bin/conda" init zsh >/dev/null 2>&1
fi

print_success "Conda initialized successfully!"

# Cleanup
print_step "Cleaning up..."
rm -f "$DOWNLOAD_DIR/$INSTALLER"
print_success "Cleanup completed"

# Final instructions
echo ""
echo "========================================"
echo "  ✅ Installation Complete!"
echo "========================================"
echo ""
print_info "Conda has been installed to: $INSTALL_DIR"
echo ""
print_step "Next Steps:"
echo ""
echo "1. Reload your shell configuration:"
echo -e "   ${GREEN}source ~/.bashrc${NC}"
echo ""
echo "2. Verify conda is working:"
echo -e "   ${GREEN}conda --version${NC}"
echo ""
echo "3. Run the Robot Vision Toolkit setup:"
echo -e "   ${GREEN}bash setup_all_in_one.sh${NC}"
echo ""
print_warning "Note: If you're using a different shell (zsh, fish, etc.), you may need to"
print_warning "      restart your terminal or source the appropriate configuration file."
echo ""
