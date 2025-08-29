"""
Fix Data Leakage Issue - Regenerate Clean Datasets
Ensures no semiprime N appears in both train and test sets.
"""

import numpy as np
import pandas as pd
import os
import json
from typing import List, Tuple, Set
import random
from sympy import isprime, factorint

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)


def generate_prime_in_range(min_bits: int, max_bits: int) -> int:
    """Generate a prime number in the specified bit range."""
    if max_bits <= 1:
        # Handle small cases
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        return random.choice([p for p in small_primes if p.bit_length() <= max_bits])
    
    min_val = 2**(min_bits-1) if min_bits > 1 else 2
    max_val = 2**max_bits - 1
    
    attempts = 0
    while attempts < 10000:
        candidate = random.randint(min_val, max_val)
        if candidate % 2 == 0:
            candidate += 1
        if isprime(candidate):
            return candidate
        attempts += 1
    
    # Fallback: return a known prime in range
    return next(p for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] if min_val <= p <= max_val)


def generate_clean_semiprimes(total_samples: int, max_value: int) -> Tuple[List[int], List[int], Set[int]]:
    """Generate semiprimes ensuring no duplicates."""
    
    semiprimes = []
    factors = []
    seen_N = set()
    
    print(f"Generating {total_samples} unique semiprimes up to {max_value:,}...")
    
    # Calculate reasonable bit ranges
    max_bits = max_value.bit_length()
    target_factor_bits = max_bits // 2 + 1
    
    attempts = 0
    while len(semiprimes) < total_samples and attempts < total_samples * 10:
        attempts += 1
        
        # Generate two primes
        if max_bits <= 8:
            # For small ranges, be more flexible
            p = generate_prime_in_range(1, target_factor_bits + 2)
            q = generate_prime_in_range(1, target_factor_bits + 2)
        else:
            p = generate_prime_in_range(max(1, target_factor_bits - 2), target_factor_bits + 2)
            q = generate_prime_in_range(max(1, target_factor_bits - 2), target_factor_bits + 2)
        
        N = p * q
        
        # Check constraints
        if N > max_value:
            continue
        if N in seen_N:
            continue
        
        # Store smaller prime as factor (following papers)
        smaller_p = min(p, q)
        
        semiprimes.append(N)
        factors.append(smaller_p)
        seen_N.add(N)
        
        if len(semiprimes) % 1000 == 0:
            print(f"  Generated {len(semiprimes)}/{total_samples} unique semiprimes...")
    
    if len(semiprimes) < total_samples:
        print(f"  Warning: Only generated {len(semiprimes)} out of {total_samples} requested samples")
    
    return semiprimes, factors, seen_N


def create_clean_train_test_split(semiprimes: List[int], factors: List[int], test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/test split ensuring no N overlap."""
    
    # Create combined data
    data = list(zip(semiprimes, factors))
    random.shuffle(data)
    
    # Split without overlap (N values are already unique from generation)
    split_idx = int(len(data) * (1 - test_ratio))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Create DataFrames
    train_df = pd.DataFrame(train_data, columns=['N', 'p'])
    test_df = pd.DataFrame(test_data, columns=['N', 'p'])
    
    # Verify no overlap
    train_Ns = set(train_df['N'])
    test_Ns = set(test_df['N'])
    overlap = train_Ns.intersection(test_Ns)
    
    if overlap:
        raise ValueError(f"Data leakage detected: {len(overlap)} overlapping N values!")
    
    print(f"  Clean split: {len(train_df)} train / {len(test_df)} test (no overlap)")
    
    return train_df, test_df


def regenerate_dataset(scale_config: dict):
    """Regenerate a single dataset with no leakage."""
    
    name = scale_config["name"]
    print(f"\n=== REGENERATING {name.upper()} DATASET ===")
    
    # Generate clean semiprimes
    semiprimes, factors, seen_N = generate_clean_semiprimes(
        scale_config["num_samples"], 
        scale_config["max_value"]
    )
    
    if len(semiprimes) == 0:
        print(f"Failed to generate data for {name}")
        return
    
    # Create clean train/test split
    train_df, test_df = create_clean_train_test_split(semiprimes, factors)
    
    # Calculate statistics
    max_N = max(semiprimes)
    max_p = max(factors)
    max_N_bits = max_N.bit_length()
    max_p_bits = max_p.bit_length()
    
    # Create metadata
    metadata = {
        "dataset_name": name,
        "description": scale_config["description"],
        "num_samples": len(semiprimes),
        "max_value": max_N,
        "max_semiprime_bits": max_N_bits,
        "max_factor_bits": max_p_bits,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "format": "clean_no_leakage",
        "columns": ["N", "p"],
        "generation_config": scale_config,
        "data_leakage_verified": "NONE"
    }
    
    # Save files
    os.makedirs("data", exist_ok=True)
    
    train_df.to_csv(f"data/{name}_train.csv", index=False)
    test_df.to_csv(f"data/{name}_test.csv", index=False)
    
    with open(f"data/{name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved: data/{name}_train.csv ({len(train_df)} samples)")
    print(f"  Saved: data/{name}_test.csv ({len(test_df)} samples)")
    print(f"  Saved: data/{name}_metadata.json")
    print(f"  Max N: {max_N:,} ({max_N_bits} bits)")
    print(f"  Max p: {max_p:,} ({max_p_bits} bits)")


def main():
    """Regenerate all datasets with no data leakage."""
    
    print("FIXING DATA LEAKAGE - REGENERATING CLEAN DATASETS")
    print("=" * 60)
    print("This will create new train/test splits with NO overlapping N values")
    print("All previous model results will need to be re-evaluated")
    print()
    
    # Define scales (matching original but with clean generation)
    scales = [
        {
            "name": "tiny", 
            "num_samples": 200, 
            "description": "N < 1,000 (Testing scale)", 
            "max_value": 1000
        },
        {
            "name": "small", 
            "num_samples": 2000, 
            "description": "N < 10,000 (Murat scale)", 
            "max_value": 10000
        },
        {
            "name": "medium", 
            "num_samples": 20000, 
            "description": "N < 100,000 (Murat scale)", 
            "max_value": 100000
        }
    ]
    
    # Regenerate each dataset
    for scale in scales:
        try:
            regenerate_dataset(scale)
        except Exception as e:
            print(f"Error regenerating {scale['name']}: {e}")
            continue
    
    print(f"\nDATASET REGENERATION COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("1. Run: python evaluate_models.py  # Verify no data leakage")
    print("2. Retrain models with clean data")
    print("3. Get honest performance metrics for UROP")


if __name__ == "__main__":
    main()