# Feature Extraction Methods for RSA Factorization Models

## Overview

This document explains the feature extraction methods used across all implemented models in our RSA semiprime factorization research. Each model uses different input representations, ranging from simple binary vectors to sophisticated 125-dimensional mathematical feature vectors.

## 1. Binary LSTM (Baseline Model)

### Feature Extraction Method
**Input Representation**: Pure binary vectors following Murat et al. methodology

```python
def convert_to_binary_vectors(N_values, p_values, max_N_bits=20, max_p_bits=10):
    """Convert semiprimes and factors to binary vectors."""
    for N, p in zip(N_values, p_values):
        # Convert N to binary (input)
        N_binary = format(N, f'0{max_N_bits}b')
        X_binary.append([int(bit) for bit in N_binary])
        
        # Convert p to binary (target output)
        p_binary = format(p, f'0{max_p_bits}b')
        y_binary.append([int(bit) for bit in p_binary])
```

### Feature Vector Structure
- **Input Size**: 14 bits (for N < 10,000)
- **Output Size**: 7 bits (for factors < 128)
- **Example**: N = 77 → [0,0,0,0,0,0,0,1,0,0,1,1,0,1] (14-bit)
- **Target**: p = 7 → [0,0,0,0,1,1,1] (7-bit)

### Processing Method
```python
# LSTM processes each bit as a time step
x = x.unsqueeze(2)  # (batch, bits, 1)
# Sequence: [0] → [0,0] → [0,0,0] → ... → [0,0,0,1,0,0,1,1,0,1]
```

## 2. Enhanced Transformer Model (125-Dimensional Features)

### Feature Extraction Method
**Input Representation**: Comprehensive mathematical feature engineering

The `FeatureEngineer` class extracts 125-dimensional vectors incorporating:

#### 2.1 Binary Representation (30 features)
```python
# Basic binary representation (30 bits padded)
binary_repr = format(n, 'b')
binary_features = [int(b) for b in binary_repr]
while len(binary_features) < 30:
    binary_features.insert(0, 0)  # Pad with zeros
```

#### 2.2 Modular Residues (5 features)
```python
def extract_modular_residues(n, moduli=[3, 5, 7, 11, 13]):
    """Extract N mod m for mathematical analysis."""
    return [float(n % m) for m in moduli]
```

#### 2.3 Hamming Weight (1 feature)
```python
def calculate_hamming_weight(n):
    """Sum of binary digits - indicates number density."""
    return float(bin(n).count('1'))
```

#### 2.4 ECPP-Based Features (45 features)
**Elliptic Curve Primality Proving inspired features:**

```python
class ECPPFeatureExtractor:
    def extract_eccp_signature(self, n):
        features = []
        
        # Modular residues with small primes (10 features)
        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            features.append(n % p)
        
        # Quadratic residues - Legendre symbols (5 features)
        for p in [3, 5, 7, 11, 13]:
            legendre = pow(n, (p-1)//2, p) if n % p != 0 else 0
            features.append(legendre)
        
        # Elliptic curve properties (15 features)
        features.extend(self._extract_curve_properties(n))
        
        # Complex multiplication properties (10 features)  
        features.extend(self._extract_cm_properties(n))
        
        # Point count estimates (3 features)
        features.extend(self._extract_point_count_estimates(n))
        
        # Binary properties (2 features)
        features.append(bin(n).count('1'))  # Hamming weight
        features.append(n.bit_length())     # Bit length
```

#### 2.5 GNFS-Inspired Features (44 features)
**General Number Field Sieve inspired characteristics:**

```python
class GNFSInspiredFeatures:
    def extract_gnfs_features(self, n):
        # Smoothness indicators (20 features)
        smoothness = self._calculate_smoothness_indicators(n)
        
        # Factor base analysis (15 features) 
        factor_base = self._analyze_factor_base_properties(n)
        
        # Polynomial properties (9 features)
        polynomial = self._extract_polynomial_characteristics(n)
```

### Complete 125-Dimensional Vector Structure
```
Feature Vector = [
    Binary[30] +           # Basic binary representation
    ModRes[5] +           # Modular residues  
    Hamming[1] +          # Hamming weight
    ECPP[45] +            # Elliptic curve features
    GNFS[44]              # Number field sieve features
] = 125 dimensions total
```

## 3. Dual-Loss LSTM Model

### Feature Extraction Method
**Input Representation**: Same as Binary LSTM but with dual outputs

```python
class DualOutputDataset:
    def __init__(self, N_values, p_values, q_values, max_N_bits=20, max_factor_bits=10):
        # Calculate max bits needed for BOTH p and q
        max_p = max(p_values)
        max_q = max(q_values) 
        max_factor_bits = max(int(max_p).bit_length(), int(max_q).bit_length())
        
        for N, p, q in zip(N_values, p_values, q_values):
            # Input: N as binary (same as Binary LSTM)
            N_binary = format(N, f'0{max_N_bits}b')
            
            # Output 1: p as binary 
            p_binary = format(p, f'0{max_factor_bits}b')
            
            # Output 2: q as binary (same bit length as p)
            q_binary = format(q, f'0{max_factor_bits}b')
```

### Key Difference from Binary LSTM
- **Same input processing** as Binary LSTM
- **Dual output heads** predicting both p and q simultaneously
- **Dynamic bit sizing** based on max(p_bits, q_bits) to handle larger factors

## 4. Generative Adversarial Network (GAN)

### Feature Extraction Method
**Input Representation**: Enhanced features + noise vector

```python
class PrimeFactorGenerator:
    def forward(self, semiprime_features, noise=None):
        # Input: Enhanced 125D features (same as Transformer)
        # + Random noise vector (128 dimensions)
        
        if noise is None:
            noise = torch.randn(batch_size, 128, device=device)
            
        # Concatenate semiprime features with noise
        x = torch.cat([semiprime_features, noise], dim=1)
        # Total input: 125 + 128 = 253 dimensions
```

### Generator Processing
```python
# Generator architecture processes combined input
input_layer: 253 → 256 (with LayerNorm)
hidden_layers: 256 → 512 → 256 (with residual connections)
constraint_layer: 256 → 128 (mathematical constraints)
output_layer: 128 → 7 (factor bits with sigmoid)
```

### Discriminator Input
```python
class PrimeFactorDiscriminator:
    def forward(self, semiprime_features, factor_bits):
        # Input: Enhanced features (125D) + Generated factor bits (7D)  
        combined_input = torch.cat([semiprime_features, factor_bits], dim=1)
        # Total: 125 + 7 = 132 dimensions
```

## Feature Extraction Comparison Summary

| Model | Input Representation | Dimensions | Key Features |
|-------|---------------------|------------|--------------|
| **Binary LSTM** | Pure binary vectors | 14 → 7 | Sequential bit processing |
| **Enhanced Transformer** | Mathematical features | 125 → 7 | Multi-head attention on math properties |
| **Dual-Loss LSTM** | Binary vectors | 14 → 9+9 | Predicts both factors simultaneously |
| **GAN** | Enhanced + noise | 125+128 → 7 | Adversarial training with noise |

## Mathematical Foundations

### Why Enhanced Features Work Better

1. **Number-Theoretic Properties**: Modular residues capture divisibility patterns
2. **Elliptic Curve Features**: Mathematical relationships from ECPP primality testing  
3. **Smoothness Indicators**: Measures how "factorable" a number appears
4. **Quadratic Residues**: Legendre symbols reveal multiplicative structure
5. **Binary Patterns**: Hamming weights and bit distributions show structural properties

### Feature Engineering Philosophy

The enhanced features move beyond simple bit patterns to capture:
- **Mathematical Structure**: Properties that classical algorithms use
- **Factorization Hints**: Patterns that suggest factor relationships  
- **Primality Indicators**: Features from primality testing literature
- **Computational Complexity**: Measures of algorithmic difficulty

This comprehensive approach enables neural networks to learn mathematical relationships that pure binary representations cannot capture effectively.