"""
Exhaustive Semiprime Data Generation - Following Murat et al. Exactly
This implements the exact methodology from Murat et al. paper:
1. Set upperbound
2. Check ALL numbers from 0 to upperbound for semiprime property
3. Convert all semiprimes to binary format
4. Save as CSV for train/test split during training
"""

import numpy as np
import pandas as pd
from sympy import factorint, isprime
import os
import json
from tqdm import tqdm
import argparse
from typing import List, Tuple


def is_semiprime(n: int) -> Tuple[bool, int]:
    """
    Check if number is a semiprime (product of exactly two primes).

    Args:
        n: Number to check

    Returns:
        Tuple of (is_semiprime, smallest_prime_factor)
    """
    if n < 4:  # Smallest semiprime is 4 = 2*2
        return False, 0

    # Factor the number
    factors = factorint(n)

    # Check if it's a semiprime (exactly 2 prime factors counting multiplicity)
    total_factors = sum(factors.values())

    if total_factors == 2:
        # Get the smallest prime factor
        smallest_prime = min(factors.keys())
        return True, smallest_prime

    return False, 0


def generate_exhaustive_semiprimes(upperbound: int) -> Tuple[List[int], List[int]]:
    """
    Generate ALL semiprimes from 0 to upperbound (Murat et al. methodology).

    Args:
        upperbound: Maximum value to check

    Returns:
        Tuple of (semiprimes, smallest_factors)
    """
    print(f"Checking ALL numbers from 0 to {upperbound:,} for semiprime property...")
    print("This is the EXACT methodology from Murat et al. paper")

    semiprimes = []
    smallest_factors = []

    # Check every single number from 0 to upperbound
    for n in tqdm(range(upperbound + 1), desc="Checking numbers"):
        is_semi, smallest_factor = is_semiprime(n)
        if is_semi:
            semiprimes.append(n)
            smallest_factors.append(smallest_factor)

    print(f"Found {len(semiprimes):,} semiprimes out of {upperbound+1:,} numbers checked")
    print(f"Semiprime density: {len(semiprimes)/(upperbound+1)*100:.2f}%")

    return semiprimes, smallest_factors


def convert_to_binary_dataset(semiprimes: List[int], factors: List[int], upperbound: int) -> pd.DataFrame:
    """
    Convert semiprimes and factors to binary format and create dataset.

    Args:
        semiprimes: List of semiprime numbers
        factors: List of corresponding smallest prime factors
        upperbound: Original upperbound (for determining bit lengths)

    Returns:
        DataFrame with binary representations
    """
    print("Converting to binary format as per Murat et al...")

    # Determine bit lengths based on upperbound (not max values)
    max_N_bits = upperbound.bit_length()
    max_p = max(factors)
    max_p_bits = max_p.bit_length()

    print(f"Using {max_N_bits} bits for N (upperbound: {upperbound:,})")
    print(f"Using {max_p_bits} bits for p (max factor: {max_p:,})")

    data = []

    for N, p in tqdm(zip(semiprimes, factors), desc="Converting to binary", total=len(semiprimes)):
        # Convert N to binary string
        N_binary = format(N, f'0{max_N_bits}b')

        # Convert p to binary string
        p_binary = format(p, f'0{max_p_bits}b')

        data.append({
            'N': N,
            'p': p,
            'q': N // p,  # Calculate q for completeness
            'N_binary': N_binary,
            'p_binary': p_binary,
            'N_bits': max_N_bits,
            'p_bits': max_p_bits
        })

    df = pd.DataFrame(data)

    print(f"Created dataset with {len(df):,} semiprime samples")
    print(f"Sample data:")
    print(df[['N', 'p', 'q', 'N_binary', 'p_binary']].head())

    return df


def save_exhaustive_dataset(df: pd.DataFrame, scale_name: str, upperbound: int):
    """Save the exhaustive dataset with metadata."""

    # Create data directory
    os.makedirs('data', exist_ok=True)

    # Save complete dataset
    dataset_path = f"data/{scale_name}_complete.csv"
    df.to_csv(dataset_path, index=False)

    # Create train/test split (80/20)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(len(df) * 0.8)

    df_train = df_shuffled[:train_size]
    df_test = df_shuffled[train_size:]

    # Save train/test splits
    train_path = f"data/{scale_name}_train.csv"
    test_path = f"data/{scale_name}_test.csv"

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    # Create metadata
    metadata = {
        "methodology": "exhaustive_enumeration_murat_et_al",
        "description": f"ALL semiprimes from 0 to {upperbound:,} (complete enumeration)",
        "scale_name": scale_name,
        "upperbound": upperbound,
        "total_numbers_checked": upperbound + 1,
        "total_semiprimes_found": len(df),
        "semiprime_density_percent": len(df) / (upperbound + 1) * 100,
        "max_N_bits": int(df['N_bits'].iloc[0]),
        "max_p_bits": int(df['p_bits'].iloc[0]),
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "largest_semiprime": int(df['N'].max()),
        "largest_factor": int(df['p'].max()),
        "columns": ["N", "p", "q", "N_binary", "p_binary", "N_bits", "p_bits"],
        "random_seed": 42,
        "train_test_split": "80/20"
    }

    metadata_path = f"data/{scale_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset saved:")
    print(f"  Complete: {dataset_path}")
    print(f"  Training: {train_path} ({len(df_train):,} samples)")
    print(f"  Testing: {test_path} ({len(df_test):,} samples)")
    print(f"  Metadata: {metadata_path}")

    return train_path, test_path, metadata_path


def verify_exhaustive_dataset(dataset_path: str):
    """Verify the exhaustive dataset integrity."""
    print(f"\nVerifying exhaustive dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Basic checks
    assert 'N' in df.columns and 'p' in df.columns, "Must have N and p columns"
    assert len(df) > 0, "Dataset cannot be empty"

    # Verify factorization for first 10 samples
    verified = 0
    for i in range(min(10, len(df))):
        N = int(df.iloc[i]['N'])
        p = int(df.iloc[i]['p'])
        q = N // p

        if N == p * q and isprime(p) and isprime(q):
            verified += 1

    print(f"[OK] Verified {verified}/10 semiprimes factor correctly")
    print(f"[OK] Dataset shape: {df.shape}")
    print(f"[OK] Max N: {df['N'].max():,}, Max p: {df['p'].max():,}")
    print(f"[OK] Binary format available: {df.columns.tolist()}")


def generate_murat_scales():
    """Generate datasets using Murat et al. exact methodology with realistic upperbounds."""

    # Murat et al. scales - exhaustive enumeration means much smaller upperbounds
    # These give dense, complete training data
    scales = [
        {
            "name": "tiny",
            "upperbound": 1000,
            "description": "ALL semiprimes from 0 to 1,000 (complete enumeration)"
        },
        {
            "name": "small",
            "upperbound": 10000,
            "description": "ALL semiprimes from 0 to 10,000 (complete enumeration)"
        },
        {
            "name": "medium",
            "upperbound": 100000,
            "description": "ALL semiprimes from 0 to 100,000 (complete enumeration)"
        },
        {
            "name": "large",
            "upperbound": 1000000,
            "description": "ALL semiprimes from 0 to 1,000,000 (complete enumeration)"
        }
    ]

    for scale in scales:
        print(f"\n{'='*80}")
        print(f"GENERATING {scale['name'].upper()} DATASET")
        print(f"Methodology: {scale['description']}")
        print(f"{'='*80}")

        # Step 1: Exhaustive enumeration (Murat et al. methodology)
        semiprimes, factors = generate_exhaustive_semiprimes(scale["upperbound"])

        # Step 2: Convert to binary format
        df = convert_to_binary_dataset(semiprimes, factors, scale["upperbound"])

        # Step 3: Save with train/test split
        train_path, test_path, metadata_path = save_exhaustive_dataset(
            df, scale["name"], scale["upperbound"]
        )

        print(f"\n{scale['name'].upper()} DATASET COMPLETE:")
        print(f"  Found {len(df):,} semiprimes from {scale['upperbound']+1:,} numbers")
        print(f"  Density: {len(df)/(scale['upperbound']+1)*100:.2f}%")
        print(f"  Training samples: {int(len(df)*0.8):,}")
        print(f"  Test samples: {len(df) - int(len(df)*0.8):,}")


def main():
    parser = argparse.ArgumentParser(description='Generate exhaustive semiprime datasets (Murat et al. methodology)')
    parser.add_argument('--scale', choices=['tiny', 'small', 'medium', 'large', 'all'],
                       default='small', help='Dataset scale to generate')
    parser.add_argument('--upperbound', type=int, help='Custom upperbound for exhaustive enumeration')
    parser.add_argument('--verify', action='store_true', help='Verify generated datasets')

    args = parser.parse_args()

    if args.upperbound:
        # Custom upperbound
        print(f"Generating custom dataset with upperbound {args.upperbound:,}")
        semiprimes, factors = generate_exhaustive_semiprimes(args.upperbound)
        df = convert_to_binary_dataset(semiprimes, factors, args.upperbound)
        save_exhaustive_dataset(df, "custom", args.upperbound)

    elif args.scale == 'all':
        # Generate all standard scales
        generate_murat_scales()

    else:
        # Generate single scale
        scale_configs = {
            "tiny": 1000,
            "small": 10000,
            "medium": 100000,
            "large": 1000000
        }

        upperbound = scale_configs[args.scale]
        print(f"Generating {args.scale} dataset with upperbound {upperbound:,}")

        semiprimes, factors = generate_exhaustive_semiprimes(upperbound)
        df = convert_to_binary_dataset(semiprimes, factors, upperbound)
        save_exhaustive_dataset(df, args.scale, upperbound)

    # Verify datasets if requested
    if args.verify:
        print(f"\n{'='*60}")
        print("VERIFYING GENERATED DATASETS")
        print(f"{'='*60}")

        if os.path.exists('data'):
            for filename in os.listdir('data'):
                if filename.endswith('_train.csv'):
                    verify_exhaustive_dataset(os.path.join('data', filename))


if __name__ == "__main__":
    main()