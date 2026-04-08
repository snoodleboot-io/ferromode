#!/bin/bash

################################################################################
# Training Pipeline Runner
# 
# Comprehensive script that:
# 1. Cleans up old training artifacts
# 2. Sets up environment and dependencies
# 3. Runs the training pipeline
# 4. Verifies the output model
#
# Usage: ./run_training.sh
################################################################################

set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/crates/ferromode/models"
MODEL_FILE="${MODELS_DIR}/lstm_predictor.safetensors"
LOG_FILE="${SCRIPT_DIR}/training.log"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

# Configuration
EPOCHS=100
BATCH_SIZE=16
LEARNING_RATE=0.0005
MIN_MODEL_SIZE=1000000  # 1 MB in bytes
MAX_MODEL_SIZE=2100000  # 2.1 MB in bytes

################################################################################
# Utility Functions
################################################################################

log_info() {
    local msg="$1"
    echo -e "${BLUE}[INFO]${NC} $msg" | tee -a "$LOG_FILE"
}

log_success() {
    local msg="$1"
    echo -e "${GREEN}[SUCCESS]${NC} $msg" | tee -a "$LOG_FILE"
}

log_warn() {
    local msg="$1"
    echo -e "${YELLOW}[WARN]${NC} $msg" | tee -a "$LOG_FILE"
}

log_error() {
    local msg="$1"
    echo -e "${RED}[ERROR]${NC} $msg" | tee -a "$LOG_FILE"
}

log_divider() {
    echo "================================================================================" | tee -a "$LOG_FILE"
}

print_header() {
    local title="$1"
    log_divider
    log_info "$title"
    log_divider
}

# Exit with error
exit_error() {
    local msg="$1"
    log_error "$msg"
    log_divider
    exit 1
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

################################################################################
# Initialization
################################################################################

init_log() {
    # Clear previous log
    > "$LOG_FILE"
    
    log_info "Training Pipeline Started"
    log_info "Timestamp: $TIMESTAMP"
    log_info "Script Directory: $SCRIPT_DIR"
    log_info "Repository Root: $REPO_ROOT"
    log_info "Models Directory: $MODELS_DIR"
    log_info "Log File: $LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

################################################################################
# Phase 1: Cleanup
################################################################################

cleanup_old_artifacts() {
    print_header "PHASE 1: Cleanup Old Artifacts"
    
    # Create models directory if it doesn't exist
    if [ ! -d "$MODELS_DIR" ]; then
        log_info "Creating models directory: $MODELS_DIR"
        mkdir -p "$MODELS_DIR"
    fi
    
    # Remove old model file
    if [ -f "$MODEL_FILE" ]; then
        log_info "Removing old model: $MODEL_FILE"
        rm -f "$MODEL_FILE"
        log_success "Old model removed"
    else
        log_info "No old model found (this is OK)"
    fi
    
    # Clean Python cache files
    log_info "Cleaning Python cache files..."
    
    # Remove __pycache__ directories
    if find "$SCRIPT_DIR" -type d -name "__pycache__" 2>/dev/null | grep -q .; then
        log_info "Removing __pycache__ directories..."
        find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        log_success "__pycache__ directories removed"
    fi
    
    # Remove .pytest_cache
    if [ -d "$SCRIPT_DIR/.pytest_cache" ]; then
        log_info "Removing .pytest_cache..."
        rm -rf "$SCRIPT_DIR/.pytest_cache"
        log_success ".pytest_cache removed"
    fi
    
    # Remove .pyc files
    if find "$SCRIPT_DIR" -type f -name "*.pyc" 2>/dev/null | grep -q .; then
        log_info "Removing .pyc files..."
        find "$SCRIPT_DIR" -type f -name "*.pyc" -delete
        log_success ".pyc files removed"
    fi
    
    # Remove .egg-info directories
    if find "$SCRIPT_DIR" -type d -name "*.egg-info" 2>/dev/null | grep -q .; then
        log_info "Removing .egg-info directories..."
        find "$SCRIPT_DIR" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
        log_success ".egg-info directories removed"
    fi
    
    # Clean torch/numpy caches if they exist
    if [ -d ~/.cache/torch ]; then
        log_warn "Found torch cache at ~/.cache/torch (keeping for reuse)"
    fi
    
    if [ -d ~/.cache/pip ]; then
        log_warn "Found pip cache at ~/.cache/pip (keeping for reuse)"
    fi
    
    # Check if .venv exists and note it's being preserved
    if [ -d "$SCRIPT_DIR/.venv" ]; then
        log_success ".venv directory preserved for reuse"
    fi
    
    echo "" | tee -a "$LOG_FILE"
    log_success "Cleanup phase completed"
    echo "" | tee -a "$LOG_FILE"
}

################################################################################
# Phase 2: Setup
################################################################################

setup_environment() {
    print_header "PHASE 2: Environment Setup"
    
    # Change to script directory
    cd "$SCRIPT_DIR" || exit_error "Failed to change to script directory"
    log_info "Working directory: $(pwd)"
    
    # Check for uv command
    if ! command_exists uv; then
        exit_error "uv package manager not found. Please install uv from https://github.com/astral-sh/uv"
    fi
    log_success "uv package manager found"
    
    # Run uv sync to ensure dependencies are installed
    log_info "Running 'uv sync' to install/update dependencies..."
    if ! uv sync 2>&1 | tee -a "$LOG_FILE"; then
        exit_error "Failed to sync dependencies with 'uv sync'"
    fi
    log_success "Dependencies synchronized"
    
    # Verify Python is available
    log_info "Verifying Python availability..."
    if ! uv run python --version 2>&1 | tee -a "$LOG_FILE"; then
        exit_error "Python is not available via uv"
    fi
    log_success "Python is available"
    
    # Verify PyTorch is available
    log_info "Verifying PyTorch installation..."
    if ! uv run python -c "import torch; print(f'PyTorch {torch.__version__} available')" 2>&1 | tee -a "$LOG_FILE"; then
        exit_error "PyTorch is not installed or not available. Please check your environment and run 'uv sync' again."
    fi
    log_success "PyTorch is available"
    
    echo "" | tee -a "$LOG_FILE"
    log_success "Setup phase completed"
    echo "" | tee -a "$LOG_FILE"
}

################################################################################
# Phase 3: Training
################################################################################

run_training() {
    print_header "PHASE 3: Training Pipeline"
    
    cd "$SCRIPT_DIR" || exit_error "Failed to change to script directory"
    
    log_info "Training Configuration:"
    log_info "  - Epochs: $EPOCHS"
    log_info "  - Batch Size: $BATCH_SIZE"
    log_info "  - Learning Rate: $LEARNING_RATE"
    log_info "  - Output Path: $MODEL_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    log_info "Starting training process..."
    log_info "Command: uv run python -m ferromode_training.train --epochs $EPOCHS --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE"
    echo "" | tee -a "$LOG_FILE"
    
    # Capture training output and log it
    if uv run python -m ferromode_training.train \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" 2>&1 | tee -a "$LOG_FILE"; then
        log_success "Training completed successfully"
    else
        exit_error "Training failed. Check $LOG_FILE for details."
    fi
    
    echo "" | tee -a "$LOG_FILE"
    log_success "Training phase completed"
    echo "" | tee -a "$LOG_FILE"
}

################################################################################
# Phase 4: Verification
################################################################################

verify_model() {
    print_header "PHASE 4: Model Verification"
    
    # Check if model file exists
    if [ ! -f "$MODEL_FILE" ]; then
        exit_error "Model file was not created at expected path: $MODEL_FILE"
    fi
    log_success "Model file found: $MODEL_FILE"
    
    # Check file size
    local file_size=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat -c%s "$MODEL_FILE" 2>/dev/null)
    
    if [ -z "$file_size" ]; then
        exit_error "Could not determine model file size"
    fi
    
    log_info "Model file size: $file_size bytes ($(numfmt --to=iec-i --suffix=B $file_size 2>/dev/null || echo "$((file_size / 1000)) KB"))"
    
    # Verify file size is reasonable
    if [ "$file_size" -lt "$MIN_MODEL_SIZE" ] || [ "$file_size" -gt "$MAX_MODEL_SIZE" ]; then
        log_warn "Model file size is outside expected range (1-2 MB)"
        log_warn "Expected: $MIN_MODEL_SIZE - $MAX_MODEL_SIZE bytes"
        log_warn "Actual: $file_size bytes"
        log_warn "This may indicate an incomplete or corrupted training run"
    else
        log_success "Model file size is within acceptable range"
    fi
    
    # Verify file is readable
    if [ ! -r "$MODEL_FILE" ]; then
        exit_error "Model file exists but is not readable: $MODEL_FILE"
    fi
    log_success "Model file is readable"
    
    # Display file details
    local file_time=$(stat -f "%Sm -t %Y-%m-%d %H:%M:%S" "$MODEL_FILE" 2>/dev/null || stat -c %y "$MODEL_FILE" | cut -d. -f1 2>/dev/null)
    log_info "File creation time: $file_time"
    
    echo "" | tee -a "$LOG_FILE"
    log_success "Verification phase completed"
    echo "" | tee -a "$LOG_FILE"
}

################################################################################
# Final Status
################################################################################

print_final_status() {
    print_header "TRAINING PIPELINE COMPLETED SUCCESSFULLY"
    
    log_success "All phases completed without errors"
    echo "" | tee -a "$LOG_FILE"
    log_info "📊 Model Location: $MODEL_FILE"
    log_info "📝 Training Log: $LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    if [ -f "$MODEL_FILE" ]; then
        local file_size=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat -c%s "$MODEL_FILE" 2>/dev/null)
        log_info "📦 Model Size: $file_size bytes"
    fi
    
    echo "" | tee -a "$LOG_FILE"
    log_info "Summary of operations:"
    log_info "  ✓ Cleaned up old training artifacts"
    log_info "  ✓ Synchronized dependencies (uv sync)"
    log_info "  ✓ Verified PyTorch installation"
    log_info "  ✓ Ran training with $EPOCHS epochs"
    log_info "  ✓ Verified model file exists and is valid"
    echo "" | tee -a "$LOG_FILE"
    
    log_divider
    log_success "Ready for deployment or further evaluation"
    log_divider
}

################################################################################
# Error Handling & Troubleshooting
################################################################################

print_troubleshooting_help() {
    local error_type="$1"
    
    log_divider
    log_warn "Troubleshooting suggestions:"
    log_divider
    echo "" | tee -a "$LOG_FILE"
    
    case "$error_type" in
        "uv")
            log_warn "1. Ensure uv is installed: https://github.com/astral-sh/uv"
            log_warn "2. Update uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
            log_warn "3. Clear uv cache: rm -rf ~/.cache/uv"
            ;;
        "pytorch")
            log_warn "1. Check PyTorch installation in pyproject.toml"
            log_warn "2. Run 'uv sync' to reinstall: uv sync --fresh"
            log_warn "3. Check Python version compatibility: uv run python --version"
            ;;
        "training")
            log_warn "1. Check ferromode_training module exists: ls -la ferromode_training/"
            log_warn "2. Verify training parameters are valid"
            log_warn "3. Check available disk space: df -h"
            log_warn "4. Check memory availability: free -h"
            log_warn "5. Review detailed log: cat $LOG_FILE"
            ;;
        "model")
            log_warn "1. Verify training completed without errors in log"
            log_warn "2. Check output path is writable: ls -la $MODELS_DIR"
            log_warn "3. Check available disk space: df -h $MODELS_DIR"
            ;;
    esac
    
    echo "" | tee -a "$LOG_FILE"
    log_warn "For more details, examine: $LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

################################################################################
# Main Execution
################################################################################

main() {
    init_log
    
    # Run cleanup phase
    if ! cleanup_old_artifacts; then
        print_troubleshooting_help "cleanup"
        exit_error "Cleanup phase failed"
    fi
    
    # Run setup phase
    if ! setup_environment; then
        print_troubleshooting_help "setup"
        exit_error "Setup phase failed"
    fi
    
    # Run training phase
    if ! run_training; then
        print_troubleshooting_help "training"
        exit_error "Training phase failed"
    fi
    
    # Run verification phase
    if ! verify_model; then
        print_troubleshooting_help "model"
        exit_error "Verification phase failed"
    fi
    
    # Print final status
    print_final_status
    
    exit 0
}

# Trap errors and show cleanup help
trap 'print_troubleshooting_help "error"; exit 1' ERR

# Run main function
main "$@"
