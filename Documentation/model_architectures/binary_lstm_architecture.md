# Binary LSTM Architecture Specification

## Model Overview
**Based on**: Murat et al. (2020) "Integer Prime Factorization with Deep Learning"  
**Architecture**: Three-layer LSTM with LayerNorm (modified for small-batch stability)  
**Input**: 14-bit binary representation of semiprimes  
**Output**: 7-bit binary representation of smallest prime factor  

## Architecture Diagram

```
Input: N (14 bits)
    ↓
[0,1,0,1,1,0,1,0,0,1,1,0,1,0] → Reshape to (batch, 14, 1)
    ↓
┌─────────────────────────────────────────────────┐
│ LSTM Layer 1: input_size=1, hidden_size=128    │
│ + LayerNorm(128)                                │  
│ + Dropout(0.3)                                 │
└─────────────────────────────────────────────────┘
    ↓ (take last output: [:, -1, :])
┌─────────────────────────────────────────────────┐
│ LSTM Layer 2: input_size=128, hidden_size=256  │
│ + LayerNorm(256)                                │
│ + Dropout(0.3)                                 │  
└─────────────────────────────────────────────────┘
    ↓ (take last output: [:, -1, :])
┌─────────────────────────────────────────────────┐
│ LSTM Layer 3: input_size=256, hidden_size=512  │
│ + LayerNorm(512)                                │
│ + Dropout(0.3)                                 │
└─────────────────────────────────────────────────┘
    ↓ (take last output: [:, -1, :])  
┌─────────────────────────────────────────────────┐
│ Dense Layer 1: 512 → 128                       │
│ + LayerNorm(128)                                │
│ + ReLU + Dropout(0.3)                          │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Dense Layer 2: 128 → 100                       │  
│ + LayerNorm(100)                                │
│ + ReLU + Dropout(0.4)                          │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Output Layer: 100 → 7                          │
│ + Sigmoid (binary classification)               │
└─────────────────────────────────────────────────┘
    ↓
Output: p (7 bits) [0,0,0,0,1,1,1]
```

## Detailed Layer Specifications

### LSTM Layers
```python
# Layer 1: Bit-level sequence processing
self.lstm1 = nn.LSTM(
    input_size=1,           # Each bit processed individually
    hidden_size=128,        # Hidden state dimension
    batch_first=True,       # (batch, seq, features) format
    dropout=0.0            # No internal dropout (handled separately)
)
self.ln1 = nn.LayerNorm(128)    # Stable normalization for small batches
self.dropout1 = nn.Dropout(0.3)

# Layer 2: Pattern aggregation  
self.lstm2 = nn.LSTM(
    input_size=128,         # From previous LSTM output
    hidden_size=256,        # Increased capacity for patterns
    batch_first=True
)
self.ln2 = nn.LayerNorm(256)
self.dropout2 = nn.Dropout(0.3)

# Layer 3: High-level feature extraction
self.lstm3 = nn.LSTM(
    input_size=256,         # From previous LSTM output  
    hidden_size=512,        # Maximum capacity
    batch_first=True
)
self.ln3 = nn.LayerNorm(512)
self.dropout3 = nn.Dropout(0.3)
```

### Dense Layers
```python  
# Dense Layer 1: Feature compression
self.dense1 = nn.Linear(512, 128)
self.ln4 = nn.LayerNorm(128)
self.dropout4 = nn.Dropout(0.3)

# Dense Layer 2: Further compression  
self.dense2 = nn.Linear(128, 100)
self.ln5 = nn.LayerNorm(100)
self.dropout5 = nn.Dropout(0.4)    # Higher dropout before output

# Output Layer: Binary prediction
self.output = nn.Linear(100, 7)     # 7-bit factor output
```

## Forward Pass Implementation

```python
def forward(self, x):
    # Input shape: (batch_size, 14)
    # Reshape for LSTM: (batch_size, 14, 1) 
    x = x.unsqueeze(2)
    
    # LSTM Layer 1
    x, _ = self.lstm1(x)           # (batch, 14, 128)
    x = self.ln1(x[:, -1, :])      # Take last timestep: (batch, 128)
    x = self.dropout1(x)
    
    # LSTM Layer 2 (need to add sequence dimension back)
    x = x.unsqueeze(1)             # (batch, 1, 128) 
    x, _ = self.lstm2(x)           # (batch, 1, 256)
    x = self.ln2(x[:, -1, :])      # (batch, 256)
    x = self.dropout2(x)
    
    # LSTM Layer 3
    x = x.unsqueeze(1)             # (batch, 1, 256)
    x, _ = self.lstm3(x)           # (batch, 1, 512)
    x = self.ln3(x[:, -1, :])      # (batch, 512)
    x = self.dropout3(x)
    
    # Dense layers with ReLU activation
    x = torch.relu(self.dense1(x))  # (batch, 128)
    x = self.ln4(x)
    x = self.dropout4(x)
    
    x = torch.relu(self.dense2(x))  # (batch, 100)  
    x = self.ln5(x)
    x = self.dropout5(x)
    
    # Binary output with sigmoid
    x = torch.sigmoid(self.output(x))  # (batch, 7)
    return x
```

## Model Parameters

### Parameter Count Breakdown
```
LSTM Layer 1: 1 → 128
- Parameters: 4 * (1 + 128 + 1) * 128 = 66,560

LSTM Layer 2: 128 → 256  
- Parameters: 4 * (128 + 256 + 1) * 256 = 394,240

LSTM Layer 3: 256 → 512
- Parameters: 4 * (256 + 512 + 1) * 512 = 1,574,912

LayerNorm layers: 
- ln1: 128 * 2 = 256
- ln2: 256 * 2 = 512  
- ln3: 512 * 2 = 1,024
- ln4: 128 * 2 = 256
- ln5: 100 * 2 = 200

Dense Layer 1: 512 → 128
- Parameters: 512 * 128 + 128 = 65,664

Dense Layer 2: 128 → 100
- Parameters: 128 * 100 + 100 = 12,900

Output Layer: 100 → 7  
- Parameters: 100 * 7 + 7 = 707

Total Parameters: ≈ 2,116,231 (≈ 2.1M parameters)
```

## Training Configuration

### Hyperparameters
```python
# Optimization
optimizer = torch.optim.RMSprop(
    model.parameters(), 
    lr=0.001,                    # Learning rate from Murat et al.
    alpha=0.99,                  # RMSprop decay
    eps=1e-08,                   # Numerical stability
    weight_decay=0               # No L2 regularization
)

# Loss Function
criterion = nn.BCELoss()         # Binary Cross-Entropy

# Training
batch_size = 4                   # Small batch for dataset size
epochs = 30                      # Preliminary training
device = 'cpu'                   # Reproducible results
```

### Data Processing
```python
# Input preprocessing
def preprocess_input(N):
    """Convert semiprime to 14-bit binary vector."""
    N_binary = format(N, '014b')  # Pad to 14 bits
    return torch.FloatTensor([int(bit) for bit in N_binary])

# Target preprocessing  
def preprocess_target(p):
    """Convert prime factor to 7-bit binary vector."""
    p_binary = format(p, '07b')   # Pad to 7 bits
    return torch.FloatTensor([int(bit) for bit in p_binary])
```

## Key Modifications from Original Murat et al.

### 1. LayerNorm instead of BatchNorm
**Problem**: BatchNorm fails with small batch sizes (< 2 samples)
**Solution**: LayerNorm normalizes each sample independently
```python
# Original (Murat et al.)
self.bn1 = nn.BatchNorm1d(128)

# Our modification  
self.ln1 = nn.LayerNorm(128)
```

### 2. Improved Sequence Processing
**Enhancement**: More careful handling of LSTM sequence dimensions
```python
# Ensure proper sequence dimension for chained LSTMs
x = x.unsqueeze(1)  # Add sequence dim back after taking last output
```

### 3. Robust Dropout Configuration
**Enhancement**: Graduated dropout rates
```python
dropout_rates = [0.3, 0.3, 0.3, 0.3, 0.4]  # Higher before output layer
```

## Performance Characteristics

### Computational Complexity
- **Forward Pass**: O(batch_size × sequence_length × hidden_dimensions)
- **Memory Usage**: ~8.5MB for model parameters
- **Training Time**: ~2-3 minutes per epoch on CPU (small dataset)

### Expected Performance (Small Dataset)
- **β₀ (Exact Match)**: ~0%
- **β₁ (≤1 bit error)**: ~39-40%  
- **β₂ (≤2 bit errors)**: ~64%
- **β₃ (≤3 bit errors)**: ~78%

### Strengths and Limitations

**Strengths**:
- Faithful reproduction of Murat et al. architecture
- Stable training with LayerNorm modifications
- Sequential bit processing captures temporal patterns
- Proven architecture from literature

**Limitations**:  
- Limited to binary representations only
- No attention mechanism for long-range dependencies
- Sequential processing doesn't capture parallel bit relationships
- High parameter count relative to simple input representation