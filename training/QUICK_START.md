# Quick Start: Training Pipeline

## One-Liner

```bash
cd /home/john_aven/Documents/software/ferromode/training && ./run_training.sh
```

## What Happens

1. **Cleanup** - Removes old model and Python cache files
2. **Setup** - Installs dependencies via `uv sync` and verifies PyTorch
3. **Train** - Runs LSTM training for 100 epochs
4. **Verify** - Checks model exists and is the correct size

## Monitor Progress

```bash
# In another terminal:
tail -f training/training.log
```

## After Training Completes

The trained model is available at:
```
../crates/ferromode/models/lstm_predictor.safetensors
```

View the full log:
```bash
cat training.log
```

## Configuration

To change training parameters, edit these in `run_training.sh`:

```bash
EPOCHS=100              # Training epochs
BATCH_SIZE=16           # Batch size
LEARNING_RATE=0.0005    # Learning rate
```

Then run the script again.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `uv: command not found` | Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `PyTorch not available` | Run `uv sync --fresh` |
| `Permission denied` | Run `chmod +x run_training.sh` |
| Training fails | Check `tail -100 training.log` and disk space with `df -h` |

## Full Documentation

See `TRAINING_SCRIPT.md` for complete details, troubleshooting, and advanced usage.
