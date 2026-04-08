# Training Pipeline Script: `run_training.sh`

Complete bash script for managing the LSTM predictor model training pipeline, including cleanup, environment setup, training execution, and verification.

## Quick Start

```bash
cd /home/john_aven/Documents/software/ferromode/training
./run_training.sh
```

## What the Script Does

### Phase 1: Cleanup Old Artifacts

Removes training remnants from previous runs:

- **Old Model**: Deletes `../crates/ferromode/models/lstm_predictor.safetensors`
- **Python Cache**:
  - `__pycache__/` directories
  - `.pytest_cache/`
  - `*.pyc` files
  - `*.egg-info/` directories
- **Preserves**:
  - `.venv/` (virtual environment is reused for performance)
  - System-level caches (Torch, NumPy, pip - beneficial for reuse)

### Phase 2: Environment Setup

Prepares the Python environment:

1. **Dependency Sync**: Runs `uv sync` to install/update all dependencies
2. **PyTorch Verification**: Confirms PyTorch is available and functional
3. **Python Check**: Verifies Python is accessible via uv

### Phase 3: Training Pipeline

Executes the actual model training:

```bash
uv run python -m ferromode_training.train \
  --epochs 100 \
  --batch_size 16 \
  --learning_rate 0.0005
```

**Configuration:**
- **Epochs**: 100 (configurable in script)
- **Batch Size**: 16 samples per batch
- **Learning Rate**: 0.0005 (Adam optimizer)
- **Output**: `../crates/ferromode/models/lstm_predictor.safetensors`

All training output is logged to `training.log`.

### Phase 4: Model Verification

Validates the training output:

- ✓ Model file exists at expected path
- ✓ File size is reasonable (1-2 MB)
- ✓ File is readable
- ✓ Creation time is recorded

## Output Files

- **`training.log`**: Complete training log with timestamps and all output
- **`../crates/ferromode/models/lstm_predictor.safetensors`**: Trained model file

## Usage

### Basic Usage

```bash
./run_training.sh
```

### Configure Training Parameters

Edit these variables in the script before running:

```bash
EPOCHS=100              # Number of training epochs
BATCH_SIZE=16           # Samples per batch
LEARNING_RATE=0.0005    # Learning rate for optimizer
```

### Check Training Progress

Monitor in real-time:

```bash
# In another terminal, while training is running:
tail -f training.log
```

### View Complete Log

After training completes:

```bash
cat training.log
```

## Error Handling & Troubleshooting

### Missing `uv` Command

**Error**: `uv package manager not found`

**Solutions:**
1. Install uv: https://github.com/astral-sh/uv
2. Or use the standalone installer:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

### PyTorch Not Available

**Error**: `PyTorch is not installed or not available`

**Solutions:**
1. Clear uv cache and resync:
   ```bash
   uv sync --fresh
   ```
2. Verify Python version (3.9+):
   ```bash
   uv run python --version
   ```
3. Check `pyproject.toml` for correct PyTorch version

### Training Failed

**Error**: `Training failed. Check training.log for details.`

**Troubleshooting:**
1. Review detailed log:
   ```bash
   tail -100 training.log
   ```
2. Check system resources:
   ```bash
   # Memory
   free -h
   
   # Disk space
   df -h
   
   # CPU usage
   top -bn1 | head -20
   ```
3. Ensure training module is available:
   ```bash
   ls -la ferromode_training/
   ```

### Model File Not Created

**Error**: `Model file was not created at expected path`

**Troubleshooting:**
1. Check directory permissions:
   ```bash
   ls -la ../crates/ferromode/models/
   ```
2. Verify disk space:
   ```bash
   df -h ../crates/ferromode/
   ```
3. Review training.log for training errors

## Script Features

### Color-Coded Output

- 🔵 `[INFO]` - Informational messages (blue)
- 🟢 `[SUCCESS]` - Successful completions (green)
- 🟡 `[WARN]` - Warnings (yellow)
- 🔴 `[ERROR]` - Errors (red)

### Comprehensive Logging

All output is both:
- Printed to console for real-time feedback
- Logged to `training.log` for historical reference

### Phase-Based Structure

```
PHASE 1: Cleanup Old Artifacts
PHASE 2: Environment Setup
PHASE 3: Training Pipeline
PHASE 4: Model Verification
```

### Progress Indicators

- Clear dividers between phases
- Status messages for each operation
- Final summary with model details

## File Paths

| File | Location | Purpose |
|------|----------|---------|
| Script | `training/run_training.sh` | Main training orchestrator |
| Log | `training/training.log` | Complete execution log |
| Model | `crates/ferromode/models/lstm_predictor.safetensors` | Trained LSTM model |
| Module | `training/ferromode_training/` | Training code package |

## Dependencies

- **uv**: Package manager for Python
- **Python 3.9+**: Runtime
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Other**: See `pyproject.toml` for full list

## Performance Notes

### First Run
- Takes longer due to dependency installation
- Virtual environment setup
- Initial PyTorch cache building

### Subsequent Runs
- Faster due to:
  - Reused `.venv/`
  - Cached dependencies
  - Cached PyTorch libraries

### Training Duration
- Typically 5-30 minutes depending on:
  - System CPU/GPU
  - Dataset size
  - Batch size
  - Number of epochs

## Exit Codes

- **`0`**: Success - all phases completed
- **`1`**: Failure - one or more phases failed

## Advanced Usage

### Run with Custom Epochs

Edit the script and change:
```bash
EPOCHS=50  # Faster training
```

Then run:
```bash
./run_training.sh
```

### Monitor with Log Streaming

In one terminal:
```bash
./run_training.sh
```

In another terminal:
```bash
tail -f training.log
```

### Check Model After Training

```bash
# View file details
ls -lh ../crates/ferromode/models/lstm_predictor.safetensors

# Test model loading (if available)
uv run python -c "import torch; m = torch.load('...'); print('Model loaded successfully')"
```

## Cleanup Between Runs

The script automatically cleans up old artifacts. To manually clean:

```bash
# Remove old model
rm -f ../crates/ferromode/models/lstm_predictor.safetensors

# Remove Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

# Clear logs
rm -f training.log
```

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
name: Train Model
on: [push]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install uv
        uses: astral-sh/setup-uv@v1
      
      - name: Run training
        working-directory: training
        run: ./run_training.sh
      
      - name: Upload model
        uses: actions/upload-artifact@v3
        with:
          name: trained-model
          path: crates/ferromode/models/lstm_predictor.safetensors
      
      - name: Upload logs
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: training-logs
          path: training/training.log
```

## Support & Debugging

For detailed debugging:

1. **Enable verbose output** in training module (if available)
2. **Check training.log** - most issues are documented there
3. **Verify dependencies**: `uv pip list`
4. **Test PyTorch**: `uv run python -c "import torch; print(torch.__version__)"`

## Related Files

- `pyproject.toml` - Dependency definitions
- `ferromode_training/train.py` - Training implementation
- `SETUP.md` - Initial setup instructions
- `README.md` - Project overview
