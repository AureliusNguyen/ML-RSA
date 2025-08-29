"""
Data generation script for RSA semiprime factorization training.
Generates semiprimes N and their smallest prime factors p following Murat et al. methodology.

Data Format (based on papers):
- Input: Semiprime N converted to binary representation
- Output: Smallest prime factor p converted to binary representation
"""

import numpy as np
import pandas as pd
from sympy import isprime, factorint
import random
import os
import json
from tqdm import tqdm
import argparse
from typing import List, Tuple
import pickle
from src.crypto_utils import PrimeGenerator, FeatureEngineer


def generate_semiprime_data(num_samples: int, max_value: int) -> Tuple[List[int], List[int]]:
    """
    Generate semiprimes and their smallest prime factors.
    
    Args:
        num_samples: Number of semiprimes to generate
        max_value: Maximum value for the semiprimes (e.g., 1,000,000,000)
    
    Returns:
        Tuple of (semiprimes, smallest_factors)
    """
    print(f"Generating {num_samples} semiprimes with N < {max_value:,}...")
    
    # Calculate max_bits from max_value
    max_bits = max_value.bit_length()
    prime_gen = PrimeGenerator(max_bits)
    semiprimes = []
    smallest_factors = []
    
    # Generate prime pairs and create semiprimes
    pairs_needed = num_samples * 3  # Generate extra to filter for max_value
    pairs = prime_gen.generate_prime_pairs(10, max_bits, pairs_needed)
    
    for p, q in tqdm(pairs, desc="Creating semiprimes"):
        if len(semiprimes) >= num_samples:
            break
            
        if p > q:
            p, q = q, p  # Ensure p <= q
        
        N = p * q
        
        # Verify N is within our target range
        if N < max_value:
            semiprimes.append(N)
            smallest_factors.append(p)  # p is the smallest factor
    
    return semiprimes, smallest_factors


def create_simple_table_dataset(semiprimes: List[int], factors: List[int]) -> pd.DataFrame:
    """
    Create simple table dataset with N and p columns as per research papers.
    
    Args:
        semiprimes: List of semiprime numbers N
        factors: List of smallest prime factors p
    
    Returns:
        DataFrame with 'N' and 'p' columns
    """
    print("Creating simple table dataset with N and p columns...")
    
    df = pd.DataFrame({
        'N': semiprimes,
        'p': factors
    })
    
    return df


def generate_enhanced_features_dataset(semiprimes: List[int], factors: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate enhanced features using ECPP and GNFS methods.
    
    Returns:
        Tuple of (X_enhanced, y_binary) 
    """
    print("Generating enhanced mathematical features...")
    
    feature_engineer = FeatureEngineer()
    X_enhanced = []
    y_binary = []
    
    # Calculate max factor bits
    max_factor = max(factors)
    max_factor_bits = max_factor.bit_length()
    
    for N, p in tqdm(zip(semiprimes, factors), desc="Extracting features", total=len(semiprimes)):
        # Extract comprehensive features for semiprime
        features = feature_engineer.extract_all_features(N)
        X_enhanced.append(features)
        
        # Convert smallest factor to binary (same as basic approach)
        p_binary = format(p, f'0{max_factor_bits}b')
        y_binary.append([int(bit) for bit in p_binary])
    
    return np.array(X_enhanced), np.array(y_binary, dtype=np.int8)


def save_dataset(data_dict: dict, save_path: str, metadata: dict):
    """Save dataset with metadata."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save main data
    np.savez_compressed(save_path, **data_dict)
    
    # Save metadata
    metadata_path = save_path.replace('.npz', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to: {save_path}")
    print(f"Metadata saved to: {metadata_path}")


def create_train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create train/test split."""
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)
    
    split_idx = int(num_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def generate_multiple_datasets(base_config: dict):
    """Generate datasets for different scales as in the papers."""
    
    # Match both papers' scales: Murat (1K, 10K, 100K, 1M) + Nene extension (100M)
    # Adjusted sample sizes to be realistic for small ranges
    scales = [
        {"name": "tiny", "num_samples": 200, "max_bits": 10, "description": "N < 1,000 (Murat baseline)", "max_value": 1000},
        {"name": "small", "num_samples": 2000, "max_bits": 14, "description": "N < 10,000 (Murat scale)", "max_value": 10000},
        {"name": "medium", "num_samples": 20000, "max_bits": 17, "description": "N < 100,000 (Murat scale)", "max_value": 100000}, 
        {"name": "large", "num_samples": 200000, "max_bits": 20, "description": "N < 1,000,000 (Murat scale)", "max_value": 1000000},
        {"name": "xlarge", "num_samples": 1000000, "max_bits": 27, "description": "N < 100,000,000 (Nene extension)", "max_value": 100000000}
    ]
    
    for scale in scales:
        print(f"\n{'='*60}")
        print(f"GENERATING {scale['name'].upper()} DATASET: {scale['description']}")
        print(f"{'='*60}")
        
        # Generate raw data
        semiprimes, factors = generate_semiprime_data(scale["num_samples"], scale["max_value"])
        
        print(f"Generated {len(semiprimes)} valid semiprimes")
        print(f"Largest semiprime: {max(semiprimes):,} ({max(semiprimes).bit_length()} bits)")
        print(f"Largest factor: {max(factors):,} ({max(factors).bit_length()} bits)")
        
        # Create simple table dataset (following papers' methodology)
        df = create_simple_table_dataset(semiprimes, factors)
        
        # Create train/test splits
        train_size = int(len(df) * (1 - base_config.get("test_split", 0.2)))
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df_train = df_shuffled[:train_size]
        df_test = df_shuffled[train_size:]
        
        # Prepare metadata
        metadata = {
            "dataset_name": scale["name"],
            "description": scale["description"],
            "num_samples": len(semiprimes),
            "max_value": scale["max_value"],
            "max_semiprime_bits": max(semiprimes).bit_length(),
            "max_factor_bits": max(factors).bit_length(),
            "train_samples": len(df_train),
            "test_samples": len(df_test),
            "format": "simple_table",
            "columns": ["N", "p"],
            "generation_config": scale
        }
        
        # Save table datasets as CSV (following paper format)
        train_path = f"data/{scale['name']}_train.csv"
        test_path = f"data/{scale['name']}_test.csv"
        
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
        
        # Also save metadata
        metadata_path = f"data/{scale['name']}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved:")
        print(f"  Training: {train_path}")
        print(f"  Testing: {test_path}")
        print(f"  Metadata: {metadata_path}")
        
        # Print dataset summary
        print(f"\nDataset Summary:")
        print(f"  Total samples: {len(df)}")
        print(f"  Training samples: {len(df_train)}")
        print(f"  Testing samples: {len(df_test)}")
        print(f"  Max semiprime: {max(semiprimes):,}")
        print(f"  Max factor: {max(factors):,}")
        print(f"  Sample data:")
        print(df.head())


def verify_dataset(dataset_path: str):
    """Verify dataset integrity for table format."""
    print(f"\nVerifying dataset: {dataset_path}")
    
    # Load CSV data
    df = pd.read_csv(dataset_path)
    metadata_path = dataset_path.replace('_train.csv', '_metadata.json').replace('_test.csv', '_metadata.json')
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Basic checks
    assert 'N' in df.columns and 'p' in df.columns, "Dataset must have N and p columns"
    assert len(df) > 0, "Dataset is empty"
    
    # Verify some semiprimes actually factor correctly
    verified_count = 0
    for i in range(min(10, len(df))):  # Check first 10
        N = int(df.iloc[i]['N'])
        p = int(df.iloc[i]['p'])
        
        if N % p == 0:  # p divides N
            q = N // p
            if isprime(p) and isprime(q) and N == p * q:
                verified_count += 1
    
    print(f"[OK] Verified {verified_count}/10 semiprimes factor correctly")
    print(f"[OK] Dataset shape: {df.shape}")
    print(f"[OK] Sample data:")
    print(df.head(3))
    print(f"[OK] Max values: N={df['N'].max():,}, p={df['p'].max():,}")
    print(f"[OK] Metadata: {metadata['num_samples']} samples, max_value={metadata.get('max_value', 'N/A'):,}")


def main():
    parser = argparse.ArgumentParser(description='Generate RSA factorization training datasets')
    parser.add_argument('--dataset', choices=['small', 'medium', 'large', 'xlarge', 'all'], 
                       default='all', help='Dataset scale to generate')
    parser.add_argument('--data-dir', default='data', help='Directory to save datasets')
    parser.add_argument('--verify', action='store_true', help='Verify generated datasets')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Configuration
    base_config = {
        "random_seed": 42,
        "test_split": 0.2,
        "data_format": "following_murat_et_al_methodology"
    }
    
    if args.dataset == 'all':
        generate_multiple_datasets(base_config)
    else:
        # Generate single dataset
        scale_configs = {
            "tiny": {"num_samples": 200, "max_value": 1000},
            "small": {"num_samples": 2000, "max_value": 10000},
            "medium": {"num_samples": 20000, "max_value": 100000},
            "large": {"num_samples": 200000, "max_value": 1000000},
            "xlarge": {"num_samples": 1000000, "max_value": 100000000}
        }
        
        config = scale_configs[args.dataset]
        print(f"Generating {args.dataset} dataset...")
        
        # Generate the dataset using the same logic as generate_multiple_datasets
        semiprimes, factors = generate_semiprime_data(config["num_samples"], config["max_value"])
        
        print(f"Generated {len(semiprimes)} valid semiprimes")
        print(f"Largest semiprime: {max(semiprimes):,} ({max(semiprimes).bit_length()} bits)")
        print(f"Largest factor: {max(factors):,} ({max(factors).bit_length()} bits)")
        
        # Create simple table dataset
        df = create_simple_table_dataset(semiprimes, factors)
        
        # Create train/test splits
        train_size = int(len(df) * (1 - base_config.get("test_split", 0.2)))
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df_train = df_shuffled[:train_size]
        df_test = df_shuffled[train_size:]
        
        # Save datasets
        train_path = f"data/{args.dataset}_train.csv"
        test_path = f"data/{args.dataset}_test.csv"
        metadata_path = f"data/{args.dataset}_metadata.json"
        
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
        
        # Save metadata
        metadata = {
            "dataset_name": args.dataset,
            "description": f"Single dataset for {args.dataset}",
            "num_samples": len(semiprimes),
            "max_value": config["max_value"],
            "max_semiprime_bits": max(semiprimes).bit_length(),
            "max_factor_bits": max(factors).bit_length(),
            "train_samples": len(df_train),
            "test_samples": len(df_test),
            "format": "simple_table",
            "columns": ["N", "p"],
            "generation_config": config
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved:")
        print(f"  Training: {train_path}")
        print(f"  Testing: {test_path}")
        print(f"  Metadata: {metadata_path}")
    
    # Verify datasets if requested
    if args.verify:
        print("\n" + "="*60)
        print("VERIFYING GENERATED DATASETS")
        print("="*60)
        
        for dataset_file in os.listdir(args.data_dir):
            if dataset_file.endswith('.csv') and ('_train.csv' in dataset_file or '_test.csv' in dataset_file):
                verify_dataset(os.path.join(args.data_dir, dataset_file))


if __name__ == "__main__":
    main()