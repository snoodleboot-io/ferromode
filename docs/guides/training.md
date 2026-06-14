# Training Guide: Differentiable EMD in Machine Learning Pipelines

Practical guidance for training neural networks that use differentiable EMD components.

## Table of Contents

1. [Problem Formulation](#problem-formulation)
2. [Data Preparation](#data-preparation)
3. [Model Architecture](#model-architecture)
4. [Training Strategies](#training-strategies)
5. [Debugging and Monitoring](#debugging-and-monitoring)
6. [Hyperparameter Tuning](#hyperparameter-tuning)

---

## Problem Formulation

### When to Use EMD in Your Model

EMD is most valuable when:

**✓ Good fit:**
- Input is a complex, non-stationary signal
- Frequency structure matters for the task
- You have domain knowledge about signal components
- Dataset size is moderate (100s-10000s of samples)
- Inference latency is not critical (~10-50ms acceptable)

**✗ Poor fit:**
- Input is high-dimensional (images, long sequences > 1000)
- Task is simple classification with obvious patterns
- Real-time inference required (< 1ms latency needed)
- Training must be very fast
- Gradient stability is critical concern

### Comparison with Alternatives

| Method | Speed | Interpretability | Learning | When to use |
|--------|-------|------------------|----------|------------|
| CNN | Fast | Low | Good | Images, stationary signals |
| Fourier | Very Fast | Medium | Poor | Frequency analysis |
| **EMD** | **Slow** | **High** | **Good** | Complex non-stationary signals |
| Wavelet | Medium | Medium | Medium | Multi-scale analysis |
| RNN | Slow | Medium | Good | Sequential dependencies |

### Integration Points in Pipelines

```
Option 1: EMD as preprocessing
Raw Signal → [EMD] → Features → [Neural Network] → Output

Option 2: EMD as intermediate layer
Raw Signal → [Conv] → [EMD] → [Dense] → Output

Option 3: EMD with feature fusion
Raw Signal → [EMD + Conv] → [Feature Fusion] → [Dense] → Output

Option 4: Multi-path architecture
Raw Signal ├→ [EMD] → [Path 1] ─┐
           └→ [Conv] → [Path 2] ─┤ [Fusion] → Output
```

Choose based on your task and domain knowledge.

---

## Data Preparation

### Signal Normalization

**Always normalize signals before EMD:**

```python
import numpy as np

def normalize_signal(signal):
    """Normalize signal to zero mean, unit variance."""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-8:  # Handle constant signals
        return signal - mean
    return (signal - mean) / std
```

**Why normalize?**
1. Prevents gradient explosion/vanishing
2. Improves Jacobian conditioning
3. Makes training more stable
4. Facilitates learning rate selection

**PyTorch example:**
```python
# Normalize in DataLoader
class NormalizingDataset(Dataset):
    def __getitem__(self, idx):
        signal, label = self.data[idx]
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        return signal, label
```

**TensorFlow example:**
```python
# Normalize as preprocessing layer
model = tf.keras.Sequential([
    tf.keras.layers.Normalization(axis=-1),  # Normalize each sample
    DifferentiableEMDLayer(max_imfs=5),
    ...
])
model.layers[0].adapt(X_train)  # Learn normalization stats
```

### Handling Variable-Length Signals

EMD requires fixed-length signals. Handle variable lengths:

```python
def pad_or_truncate(signal, target_length=256):
    """Pad with zeros or truncate to target length."""
    if len(signal) < target_length:
        padding = np.zeros(target_length - len(signal))
        return np.concatenate([signal, padding])
    else:
        return signal[:target_length]

# Or use a more sophisticated approach:
def windowed_processing(signal, window_size=256, stride=128):
    """Extract overlapping windows from long signal."""
    windows = []
    for start in range(0, len(signal) - window_size, stride):
        window = signal[start:start + window_size]
        windows.append(window)
    return np.array(windows)
```

### Batch Processing Strategies

**Strategy 1: Uniform batch size**
```python
# All signals same length
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=default_collate
)
```

**Strategy 2: Bucketing by length**
```python
# Group signals by length for efficiency
def bucket_collate(batch):
    """Sort batch by signal length."""
    batch.sort(key=lambda x: len(x[0]))
    signals = torch.stack([s[0] for s in batch])
    labels = torch.tensor([s[1] for s in batch])
    return signals, labels

dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=bucket_collate
)
```

**Strategy 3: Ragged tensors (TensorFlow)**
```python
# Handle variable-length signals naturally
dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.RaggedTensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)
dataset = dataset.ragged_batch(32)
```

### Data Augmentation with EMD

Augment training data using EMD decomposition:

```python
class EMDMixup:
    """Mix signals in EMD feature space."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.emd = DifferentiableEMD(max_imfs=3)
    
    def __call__(self, signals, labels):
        """Mix two signals via their EMD decompositions."""
        # Decompose
        imfs = self.emd(signals)  # (batch, num_imfs, len)
        
        # Mix IMFs from different signals
        batch_size = signals.shape[0]
        indices = torch.randperm(batch_size)
        
        # Mixup parameter
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix IMFs
        mixed_imfs = lam * imfs + (1 - lam) * imfs[indices]
        
        # Reconstruct (sum IMFs back)
        mixed_signals = mixed_imfs.sum(dim=1)
        mixed_labels = (labels, labels[indices], lam)
        
        return mixed_signals, mixed_labels
```

### Train/Validation/Test Split

Recommended split for EMD models:

```python
from sklearn.model_selection import train_test_split

# 1. Split into train + test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Split train into train + validation (75/25 of train = 60/20)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)

print(f"Train: {X_train.shape[0]} ({100*len(X_train)/len(X):.0f}%)")
print(f"Val:   {X_val.shape[0]} ({100*len(X_val)/len(X):.0f}%)")
print(f"Test:  {X_test.shape[0]} ({100*len(X_test)/len(X):.0f}%)")

# Create loaders
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=32,
    shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
    batch_size=32,
    shuffle=False
)
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
    batch_size=32,
    shuffle=False
)
```

---

## Model Architecture

### EMD as Preprocessing Layer

**Use case**: Quick feature extraction before classification.

```python
class EMDPreprocessor(nn.Module):
    """Extract IMF features, feed to classifier."""
    def __init__(self, num_imfs=3):
        super().__init__()
        self.emd = DifferentiableEMD(max_imfs=num_imfs)
        
        # Classifier on flattened IMFs
        feature_dim = num_imfs * 256  # Assumes 256-length signals
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # 10 classes
        )
    
    def forward(self, signals):
        imfs = self.emd(signals)  # (batch, num_imfs, 256)
        features = imfs.reshape(imfs.size(0), -1)  # Flatten
        logits = self.classifier(features)
        return logits
```

### EMD as Intermediate Layer

**Use case**: Learn frequency decomposition as part of model.

```python
class HybridEMDCNN(nn.Module):
    """Combine EMD with convolutional layers."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5)
        self.emd = DifferentiableEMD(max_imfs=3)
        self.conv2 = nn.Conv1d(3, 32, kernel_size=3)
        self.fc = nn.Linear(32 * 250, 10)
    
    def forward(self, x):
        # x: (batch, 1, 256)
        
        # Initial convolution
        x = self.conv1(x)  # (batch, 16, 252)
        x = torch.relu(x)
        
        # EMD decomposition
        x = x.squeeze(1) if x.shape[1] == 1 else x.mean(dim=1)
        imfs = self.emd(x)  # (batch, 3, 256)
        
        # Further convolution on IMFs
        x = self.conv2(imfs)  # (batch, 32, 254)
        x = torch.relu(x)
        
        # Classify
        x = x.reshape(x.size(0), -1)
        logits = self.fc(x)
        return logits
```

### Multi-EMD Architecture

**Use case**: Extract features at different decomposition depths.

```python
class MultiScaleEMD(nn.Module):
    """Extract features from multiple EMD runs."""
    def __init__(self):
        super().__init__()
        self.emd_5 = DifferentiableEMD(max_imfs=5)
        self.emd_10 = DifferentiableEMD(max_imfs=10)
        
        # Combine features from different IMF counts
        feature_dim = (5 + 10) * 256
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, signal):
        # Two different decompositions
        imfs_5 = self.emd_5(signal)  # (batch, 5, 256)
        imfs_10 = self.emd_10(signal)  # (batch, 10, 256)
        
        # Concatenate
        features = torch.cat([
            imfs_5.reshape(imfs_5.size(0), -1),
            imfs_10.reshape(imfs_10.size(0), -1)
        ], dim=1)
        
        return self.classifier(features)
```

### Feature Aggregation Strategies

```python
# Strategy 1: Simple concatenation
features = torch.cat([imfs.reshape(batch, -1) for imfs in decompositions], dim=1)

# Strategy 2: Weighted average per IMF
weights = torch.softmax(self.imf_weights, dim=0)  # Learn weights
weighted_imfs = imfs * weights.unsqueeze(0).unsqueeze(2)
features = weighted_imfs.sum(dim=1)

# Strategy 3: Statistical features per IMF
mean_imf = imfs.mean(dim=2)  # (batch, num_imfs)
var_imf = imfs.var(dim=2)  # (batch, num_imfs)
energy_imf = (imfs ** 2).sum(dim=2)  # (batch, num_imfs)
features = torch.cat([mean_imf, var_imf, energy_imf], dim=1)

# Strategy 4: Learned pooling
attention = self.attention_net(imfs)  # (batch, num_imfs, 1)
features = (imfs * attention).sum(dim=1)
```

---

## Training Strategies

### Optimization Algorithms

**Recommended optimizers:**

```python
# Adam: Works well for most cases
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8
)

# SGD with momentum: More stable but slower
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-2,
    momentum=0.9,
    nesterov=True
)

# AdamW: Adam with weight decay, often better than L2 reg
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)
```

### Learning Rates for EMD Models

EMD gradients can be more volatile than standard networks. Use lower learning rates:

```python
# Typical learning rates
standard_nn = 1e-3 to 1e-2
emd_model = 1e-4 to 1e-3  # 10x lower

# Start low and increase if training is stable
learning_rate = 1e-4  # Conservative start

# Learning rate scheduling
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,  # Reduce LR every 10 epochs
    gamma=0.5
)
```

### Gradient Clipping

Essential for EMD models due to potential gradient explosion:

```python
# After backward pass, before step
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=100.0  # Clip to [-100, 100]
)

# Or clip per parameter
torch.nn.utils.clip_grad_value_(
    model.parameters(),
    clip_value=100.0
)

optimizer.step()
```

### Regularization Techniques

```python
# 1. L2 Regularization
loss = ce_loss + 0.001 * sum(p.pow(2).sum() for p in model.parameters())

# 2. Dropout (use after EMD)
class RegularizedEMDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emd = DifferentiableEMD(max_imfs=3)
        self.dropout = nn.Dropout(0.3)  # After EMD
        self.fc = nn.Linear(3*256, 10)
    
    def forward(self, x):
        imfs = self.emd(x)
        features = imfs.reshape(imfs.size(0), -1)
        features = self.dropout(features)  # Drop IMFs
        return self.fc(features)

# 3. Batch Normalization (after flattening IMFs)
x = DifferentiableEMDLayer()(inputs)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(64)(x)

# 4. Early Stopping
def train_with_early_stopping(model, train_loader, val_loader, 
                              num_epochs=100, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader)
        
        # Validate
        val_loss = evaluate(model, val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                model.load_state_dict(torch.load('best_model.pt'))
                break
    
    return model
```

### Batch Sizes

EMD performance scales poorly with batch size. Recommendations:

```
Signal length | Recommended batch size
64            | 64-128
128           | 32-64
256           | 16-32
512           | 8-16
1024          | 4-8
```

**Practical guidance:**
```python
# Start with small batch
batch_size = 16

# Monitor GPU memory and training speed
# If memory available: increase to 32
# If training too slow: still increase (EMD is CPU-bound)
# If gradient instability: decrease to 8
```

### Training Loop Template

```python
def train_emd_model(model, train_loader, val_loader, 
                    num_epochs=50, learning_rate=1e-4):
    """Complete training loop for EMD model."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # TRAIN
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
            
            # Update
            optimizer.step()
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # VALIDATION
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate schedule
        scheduler.step()
        
        # Logging
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model, (train_losses, val_losses)
```

---

## Debugging and Monitoring

### Logging Gradient Statistics

```python
def log_gradient_stats(model, name_prefix=""):
    """Log statistics about gradients."""
    grad_means = []
    grad_stds = []
    grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            grad_means.append(grad.mean().item())
            grad_stds.append(grad.std().item())
            grad_norms.append(grad.norm().item())
    
    print(f"{name_prefix} Gradient Statistics:")
    print(f"  Mean of means: {np.mean(grad_means):.6f}")
    print(f"  Std of stds: {np.mean(grad_stds):.6f}")
    print(f"  Norm of norms: {np.mean(grad_norms):.6f}")
    
    return {
        'grad_mean': np.mean(grad_means),
        'grad_std': np.mean(grad_stds),
        'grad_norm': np.mean(grad_norms),
    }

# In training loop
for batch_x, batch_y in train_loader:
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
    loss.backward()
    
    # Monitor gradients
    stats = log_gradient_stats(model, "Batch")
    
    optimizer.step()
```

### Monitoring Jacobian Conditioning

```python
def compute_jacobian_stats(model, sample_signal):
    """Estimate Jacobian condition number."""
    emd_module = model.emd  # Assuming emd is accessible
    
    # Finite difference Jacobian estimation
    epsilon = 1e-5
    signal_len = sample_signal.shape[-1]
    jacobian = []
    
    for i in range(signal_len):
        # Forward difference
        signal_plus = sample_signal.clone()
        signal_plus[..., i] += epsilon
        output_plus = emd_module(signal_plus)
        
        signal_minus = sample_signal.clone()
        signal_minus[..., i] -= epsilon
        output_minus = emd_module(signal_minus)
        
        # Gradient w.r.t. this dimension
        grad_i = (output_plus - output_minus) / (2 * epsilon)
        jacobian.append(grad_i.flatten())
    
    jacobian = torch.stack(jacobian)
    
    # Condition number
    U, S, V = torch.svd(jacobian)
    cond_number = S[0] / (S[-1] + 1e-10)
    
    return {
        'condition_number': cond_number.item(),
        'singular_values': S.cpu().numpy(),
    }

# Check during training
if (epoch + 1) % 10 == 0:
    sample = next(iter(train_loader))[0][0:1]
    jac_stats = compute_jacobian_stats(model, sample)
    print(f"Jacobian condition number: {jac_stats['condition_number']:.2e}")
    
    if jac_stats['condition_number'] > 1e10:
        print("Warning: Jacobian is ill-conditioned!")
```

### Early Stopping Criteria

```python
class EarlyStopping:
    """Stop training when validation metric stops improving."""
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # 'min' for loss, 'max' for accuracy
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return
        
        if self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
    
    def _is_improvement(self, current_score):
        if self.mode == 'min':
            return current_score < self.best_score - self.min_delta
        else:  # 'max'
            return current_score > self.best_score + self.min_delta

# In training loop
early_stop = EarlyStopping(patience=10, mode='min')

for epoch in range(num_epochs):
    val_loss = evaluate(model, val_loader)
    early_stop(val_loss)
    
    if early_stop.should_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

### Visualization during Training

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Loss curves
axes[0, 0].plot(train_losses, label='Train')
axes[0, 0].plot(val_losses, label='Val')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Curves')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2: Gradient norms
axes[0, 1].plot(gradient_norms)
axes[0, 1].set_ylabel('Gradient Norm')
axes[0, 1].set_title('Gradient Magnitude')
axes[0, 1].grid(True)

# Plot 3: Learning rate schedule
axes[1, 0].plot(learning_rates)
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_title('LR Schedule')
axes[1, 0].grid(True)

# Plot 4: IMF energy distribution
imfs = model.emd(sample_signal)
imf_energy = (imfs ** 2).sum(dim=2).detach().cpu().numpy()
axes[1, 1].bar(range(len(imf_energy[0])), imf_energy[0])
axes[1, 1].set_ylabel('Energy')
axes[1, 1].set_title('IMF Energy Distribution')

plt.tight_layout()
plt.savefig('training_progress.png', dpi=100)
plt.show()
```

---

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Typical Range | Notes |
|-----------|----------------|-------|
| `max_imfs` | 3-10 | More = more detail, slower |
| `learning_rate` | 1e-4 to 1e-3 | 10x lower than CNN |
| `batch_size` | 8-64 | Smaller for longer signals |
| `gradient_clip` | 100-1000 | Prevent explosion |
| `dropout` | 0.2-0.5 | After EMD |
| `weight_decay` | 1e-5 to 1e-3 | L2 regularization |

### Grid Search for EMD Models

```python
from itertools import product

param_grid = {
    'max_imfs': [3, 5, 7],
    'learning_rate': [1e-4, 5e-4, 1e-3],
    'dropout': [0.2, 0.3, 0.4],
}

results = []

for params in product(*param_grid.values()):
    param_dict = dict(zip(param_grid.keys(), params))
    
    model = EMDModel(**param_dict)
    val_loss = train_and_evaluate(model, train_loader, val_loader)
    
    results.append({
        'params': param_dict,
        'val_loss': val_loss
    })
    
    print(f"Params: {param_dict} | Val Loss: {val_loss:.4f}")

# Find best
best = min(results, key=lambda x: x['val_loss'])
print(f"Best params: {best['params']}")
```

### Validation Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)

def evaluate_model(model, test_loader, device='cpu'):
    """Comprehensive evaluation."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    # Metrics
    accuracy = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average='weighted'
    )
    cm = confusion_matrix(targets, preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
```

---

**Document Version**: 2.4.0  
**Last Updated**: 2026-04-09
