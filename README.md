# RSA ML Attack: Advanced Machine Learning Approaches to RSA Cryptanalysis

This project implements new machine learning models for RSA semiprime factorization research, building on prior work by Murat et al. and Nene & Uludag with novel architectural innovations.

## 🎯 Project Overview

**Objective**: Explore ML-based approaches to RSA semiprime factorization using:
- Enhanced ECPP (Elliptic Curve Primality Proving) features
- GNFS (General Number Field Sieve) inspired characteristics  
- Advanced neural architectures (Transformers, GANs, Hybrid CNN+RNN)

**Research Context**: This work extends existing ML factorization research with mathematical insights from classical cryptanalysis methods.

## 🏗️ Architecture

### Models Implemented

1. **Baseline LSTM**: Reproduction of Murat et al.'s architecture for benchmarking
2. **Transformer Factorizer**: Multi-head attention for mathematical pattern recognition  
3. **Factorization GAN**: Adversarial training for prime factor generation
4. **Hybrid CNN+RNN**: Combined local pattern recognition and sequence modeling

### Key Innovations

- **Enhanced Feature Engineering**: ECPP-based elliptic curve features, GNFS smoothness indicators
- **Mathematical Constraints**: Built-in primality and factorization constraints
- **Multi-Scale Architecture**: CNN for local patterns + RNN for sequences + Attention for relationships

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AureliusNguyen/ML-RSA.git
cd ML-RSA/rsa_ml_attack

# Install dependencies
pip install -r requirements.txt
# or
pip install -e .
```

### Testing

Run the test suite to verify all models work:

```bash
python test_models.py
```

### Data Generation

First, generate the training datasets:

```bash
# Generate all dataset sizes (tiny, small, medium, large, xlarge)
python generate_data.py --dataset all

# Or generate specific size
python generate_data.py --dataset small
```

### Training

Train individual models with specific scripts:

```bash
# Binary LSTM (Murat et al. reproduction)
python train_binary_models.py --dataset small --epochs 30 --batch-size 4

# Dual Loss LSTM (predicts both p and q)  
python train_dual_loss.py --dataset small --epochs 30 --batch-size 4

# Enhanced Transformer (with 125D features)
python train_enhanced_models.py --dataset small --epochs 30 --batch-size 4

# GAN (adversarial factor generation)
python train_gan.py --dataset small --epochs 30 --batch-size 4
```


## 📊 Actual Results

Current performance on small dataset (N < 10,000):

| Model | β₀ (Exact Match) | β₁ (≤1 bit error) | Parameters | Status |
|-------|------------------|-------------------|------------|---------|
| Binary LSTM | 0% | ~40% | ~500K | ✅ Working |
| Dual Loss LSTM | 0% | ~35% | ~800K | ✅ Working |  
| Enhanced Transformer | 0% | **39.58%** | **3.3M** | ✅ **Best Model** |
| GAN | ~2% (exact) | N/A | ~700K | ✅ Working |

**Key Findings**:
- Enhanced Transformer achieves **39.58%** β₁ accuracy (within 1-bit error)
- **64.58%** β₂ accuracy (within 2-bit error) shows strong pattern learning
- Results significantly exceed random chance (~1.2% for 7-bit factors)
- Data leakage issues have been identified and resolved

*All models trained on clean datasets with verified train/test splits*

## 🧮 Mathematical Foundation

### ECPP Features
- Elliptic curve discriminants and j-invariants
- Complex multiplication properties  
- Hasse bound estimates for point counts
- Primality signatures based on curve arithmetic

### GNFS Features
- Murphy E-score estimates for polynomial quality
- Smoothness indicators across different bounds
- Factor base optimization heuristics
- Number field property approximations

### Constraints Enforced
- Odd number constraints (last bit = 1)
- Factorization validity checks
- Mathematical consistency across prediction heads

## 📈 Training Process

1. **Data Generation**: Create semiprimes from random prime pairs
2. **Feature Extraction**: Apply ECPP + GNFS + binary encoding
3. **Model Training**: Train with mathematical constraints and regularization
4. **Evaluation**: β-metrics (exact match and near-miss accuracies)
5. **Comparison**: Comprehensive analysis across all architectures

## 🔬 Research Applications

### Defensive Security
- Analyze RSA implementation vulnerabilities
- Guide key size recommendations
- Develop ML-resistant cryptographic practices

### Mathematical Insights  
- Understand deep patterns in prime distributions
- Explore connections between elliptic curves and factorization
- Advance number-theoretic machine learning

### Hybrid Approaches
- Combine classical algorithms with ML acceleration
- Develop quantum-classical factorization strategies
- Create adaptive cryptanalysis tools

## 📁 Repository Structure

```
ML-RSA/
├── .gitignore                   
├── README.md                    
├── Pre-Research/                # Research background
│   ├── Integer Prime Factorization with Deep Learning.pdf
│   ├── MLApproachtoIntegerSemiprimeFactorisation.pdf
│   ├── application.md 
│   └── nguy5272_UROP_Spring2020 (1).pdf
├── kaggle_testing/             # Experimental notebooks and testing
│   ├── kaggle_binary_train.py
│   └── kaggle_binary_train_fixed.py
└── rsa_ml_attack/              # Main ML implementation
    ├── src/
    │   ├── crypto_utils.py      # ECPP/GNFS feature engineering
    │   └── models/
    │       ├── baseline_lstm.py     # Murat et al. reproduction
    │       ├── transformer_factorizer.py  # Mathematical transformer
    │       ├── factorization_gan.py # Adversarial prime generation
    │       └── hybrid_cnn_rnn.py   # Multi-scale hybrid model
    ├── data/                    # Clean datasets (no data leakage)
    │   ├── small_train.csv, small_test.csv, small_metadata.json
    │   ├── medium_train.csv, medium_test.csv, medium_metadata.json
    │   └── tiny_train.csv, tiny_test.csv, tiny_metadata.json
    ├── experiments/             # Training results and saved models
    │   ├── transformer_enhanced_small/  # Best model results
    │   ├── binary_training_small/
    │   ├── dual_training_small/
    │   └── gan_training_small/
    ├── scripts/archive/         # Historical scripts
    │   └── fix_data_leakage.py  # Data leakage correction (archived)
    ├── generate_data.py         # Dataset generation script
    ├── train_binary_models.py   # Binary LSTM training
    ├── train_dual_loss.py      # Dual output LSTM training  
    ├── train_enhanced_models.py # Transformer with enhanced features
    ├── train_gan.py            # GAN training script
    ├── evaluate_models.py      # Model evaluation utilities
    ├── test_models.py          # Model verification tests
    ├── requirements.txt        # Python dependencies
    ├── setup.py               # Package installation
    └── README.md              # Project-specific documentation
```

## 🎓 Research Context

This work builds directly on:

- **Murat et al. (2020)**: "Integer Prime Factorization with Deep Learning" - LSTM baseline
- **Nene & Uludag (2022)**: "Machine Learning Approach to Integer Prime Factorisation" - Binary approaches
- **Atkin & Morain (1993)**: "Elliptic Curves and Primality Proving" - ECPP mathematical foundation

### Novel Contributions

1. **Data Leakage Detection & Resolution**: Identified and fixed critical train/test contamination issues
2. **Enhanced Feature Engineering**: 125-dimensional feature vectors using ECPP and GNFS characteristics
3. **Transformer Architecture**: Successful application of attention mechanisms to RSA factorization
4. **Robust Training Pipeline**: BatchNorm → LayerNorm conversion for stable small-batch training
5. **Comprehensive Evaluation**: β-metrics analysis showing consistent learning patterns

## ⚡ Performance Optimization

### AWS Integration
- Configured for distributed GPU training
- Automatic experiment logging and result storage
- Scalable to larger semiprime sizes

### Mathematical Constraints
- Built-in primality testing during training
- Factorization validity enforcement
- Smooth convergence through constraint regularization

## 🔍 Evaluation Metrics

- **β₀**: Exact factor match percentage (primary metric)
- **β₁-β₄**: Near-miss accuracies (1-4 bit errors allowed)
- **Mathematical Validity**: Percentage of predictions that actually factor input semiprimes
- **Training Efficiency**: Convergence speed and stability

## 📚 References

1. Murat, B., et al. "Integer prime factorization with deep learning." (2020)
2. Nene, R., & Uludag, S. "Machine learning approach to integer prime factorisation." (2022)
3. Atkin, A.O.L., & Morain, F. "Elliptic curves and primality proving." (1993)
4. Rivest, R.L., et al. "A method for obtaining digital signatures and public-key cryptosystems." (1978)

---

**⚠️ Research Disclaimer**: This project is for educational and defensive security research only. The techniques explored are intended to understand cryptographic vulnerabilities and improve security practices, not to compromise deployed systems.

**🤝 Contributing**: This research supports the cryptographic community's understanding of ML-based cryptanalysis to develop more robust security systems.