# Ferromode LSTM Training

Train boundary prediction models for Ferromode EMD decomposition.

## Quick Start (5 minutes)

### Prerequisites
- Python 3.12+ (check: `python --version`)
- pip (usually included with Python)

### Setup (2 minutes)

**Linux/macOS:**
```bash
cd training
bash setup.sh
```

**Windows:**
```cmd
cd training
setup.bat
```

Or manually:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or:
.venv\Scripts\activate.bat  # Windows

pip install -r requirements.txt
```

### Train Model (30-60 minutes)

```bash
# Linux/Mac:
bash scripts/train.sh

# Windows:
scripts\train.bat

# Or directly:
python -m training.train
```

### Validate Results (5 minutes)

```bash
python -m training.validate \
    --model ../crates/ferromode/models/lstm_predictor.onnx
```

## What Gets Generated

```
../crates/ferromode/models/
├── lstm_predictor.onnx (2.1 MB) ← Trained model (FP16 quantized)
└── lstm_predictor_fp32.onnx (4.2 MB) ← Full precision backup
```

## GPU Support (Optional)

### NVIDIA CUDA
```bash
# Install CUDA 12.1 from: https://developer.nvidia.com/cuda-downloads
pip install torch==2.2.1+cu121
python -m training.train  # Will auto-detect GPU
```

### AMD ROCm
```bash
# Install ROCm 5.7 from: https://rocmdocs.amd.com/
pip install torch==2.2.1+rocm5.7
python -m training.train  # Will auto-detect GPU
```

### Apple Silicon (Metal)
```bash
# No extra installation needed
pip install -r requirements.txt
python -m training.train  # Will auto-detect Metal
```

## Troubleshooting

**Q: "ModuleNotFoundError: No module named 'torch'"**
- A: Run: `pip install -r requirements.txt`

**Q: "CUDA out of memory"**
- A: Edit `src/training/train.py`, set `batch_size = 16`

**Q: Training very slow**
- A: Normal on CPU (60-120 min). Use GPU if available.

**Q: Model file not created**
- A: Check `training.log` for errors, ensure write permission to `crates/ferromode/models/`

See `SETUP.md` for detailed troubleshooting.

## Files

- `src/training/` - Training modules (data generation, LSTM model, training loop, validation)
- `scripts/` - Platform-specific run scripts
- `tests/` - Unit tests for training pipeline
- `.venv/` - Python virtual environment (created by you)

## Support

Questions? See `SETUP.md` for comprehensive guide.
