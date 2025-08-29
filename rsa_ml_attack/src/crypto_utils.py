"""
Cryptographic utilities for RSA semiprime generation and analysis.
Implements ECPP-based features and GNFS-inspired characteristics.
"""

import numpy as np
import sympy
from sympy import isprime, factorint, gcd
from typing import List, Tuple, Dict, Optional
import gmpy2
from gmpy2 import mpz, is_prime, random_state, mpz_random
import random


class PrimeGenerator:
    """Generate primes and semiprimes for training data."""
    
    def __init__(self, max_bits: int = 30):
        self.max_bits = max_bits
        self.random_state = random_state()
    
    def sieve_of_eratosthenes(self, limit: int) -> List[int]:
        """Generate all primes up to limit using Sieve of Eratosthenes."""
        if limit < 2:
            return []
        
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    def generate_prime_pairs(self, min_bits: int, max_bits: int, count: int) -> List[Tuple[int, int]]:
        """Generate pairs of primes for semiprime creation."""
        pairs = []
        
        for _ in range(count):
            # Generate two primes with specified bit lengths
            # Ensure both p_bits and q_bits are at least 2
            p_bits = random.randint(max(2, min_bits//2), max(2, min_bits-2))
            q_bits = max(2, max_bits - p_bits)
            
            # Adjust if q_bits is too small
            if q_bits < 2:
                p_bits = max_bits - 2
                q_bits = 2
            
            p = self._generate_prime_with_bits(p_bits)
            q = self._generate_prime_with_bits(q_bits)
            
            # Ensure p <= q for consistency
            if p > q:
                p, q = q, p
                
            pairs.append((int(p), int(q)))
        
        return pairs
    
    def _generate_prime_with_bits(self, bits: int) -> int:
        """Generate a prime with specified number of bits."""
        if bits <= 1:
            # For very small bit counts, return small primes directly
            small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
            return random.choice(small_primes)
        
        while True:
            # Generate random number with specified bits
            candidate = mpz_random(self.random_state, 2**bits)
            if bits > 1:
                candidate |= (1 << (bits - 1))  # Set MSB
            candidate |= 1  # Make odd
            
            if is_prime(candidate):
                return int(candidate)


class ECPPFeatureExtractor:
    """Extract ECPP-based features for prime characterization."""
    
    def __init__(self):
        self.small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        self.cm_discriminants = [-3, -4, -7, -8, -11, -12, -16, -19, -20, -24]
    
    def extract_ecpp_signature(self, n: int) -> List[float]:
        """
        Extract ECCP-based signature features.
        Based on elliptic curve primality proving characteristics.
        """
        # Convert to Python int to avoid numpy int64 issues
        n = int(n)
        features = []
        
        # Modular residues with small primes
        for p in self.small_primes:
            features.append(n % p)
        
        # Quadratic residues (Legendre symbols)
        for p in self.small_primes[:5]:  # Use first 5 primes
            if p != 2:
                legendre = pow(n, (p-1)//2, p) if n % p != 0 else 0
                features.append(legendre)
        
        # Enhanced ECPP features based on research
        features.extend(self._extract_curve_properties(n))
        features.extend(self._extract_cm_properties(n))
        features.extend(self._extract_point_count_estimates(n))
        
        # Binary properties
        features.append(bin(n).count('1'))  # Hamming weight
        features.append(n.bit_length())     # Bit length
        
        # Divisibility tests
        features.append(1 if n % 4 == 1 else 0)  # Form 4k+1
        features.append(1 if n % 4 == 3 else 0)  # Form 4k+3
        
        return features
    
    def _extract_curve_properties(self, n: int) -> List[float]:
        """Extract elliptic curve properties for ECPP."""
        n = int(n)  # Convert to Python int
        properties = []
        
        # Test several curve parameters
        for a in [1, 2, 3, -1, -2]:
            for b in [1, 2, 3, -1, -2]:
                if n <= 2:
                    properties.extend([0.0, 0.0, 0.0])
                    continue
                    
                try:
                    # Discriminant Δ = -16(4a³ + 27b²)
                    discriminant = -16 * (4 * a**3 + 27 * b**2)
                    properties.append(float(discriminant % n))
                    
                    # j-invariant (simplified version)
                    if (4 * a**3 + 27 * b**2) != 0:
                        j_inv = (1728 * 4 * a**3) % n
                        properties.append(float(j_inv))
                    else:
                        properties.append(0.0)
                    
                    # Curve validity check
                    properties.append(1.0 if gcd(discriminant, n) == 1 else 0.0)
                except:
                    properties.extend([0.0, 0.0, 0.0])
                    
        return properties[:15]  # Limit feature size
    
    def _extract_cm_properties(self, n: int) -> List[float]:
        """Extract Complex Multiplication discriminant properties."""
        n = int(n)  # Convert to Python int
        cm_features = []
        
        for d in self.cm_discriminants:
            try:
                # Check if n is representable by quadratic form of discriminant d
                if d % 4 == 0 or d % 4 == 1:
                    cm_features.append(float(pow(d, (n-1)//2, n) if n > 2 else 0))
                else:
                    cm_features.append(0.0)
            except:
                cm_features.append(0.0)
                
        return cm_features
    
    def _extract_point_count_estimates(self, n: int) -> List[float]:
        """Estimate point counts on elliptic curves mod n."""
        estimates = []
        
        if n <= 2:
            return [0.0] * 5
            
        try:
            # Hasse bound estimates: |#E(Fp) - (p+1)| <= 2√p
            sqrt_n = int(n**0.5)
            hasse_lower = max(0, n + 1 - 2 * sqrt_n)
            hasse_upper = n + 1 + 2 * sqrt_n
            
            estimates.extend([
                float(hasse_lower % 1000),  # Lower bound (mod 1000)
                float(hasse_upper % 1000),  # Upper bound (mod 1000)
                float(sqrt_n % 100),        # Square root estimate
                float((n + 1) % 1000),      # Central estimate
                float(abs(hasse_upper - hasse_lower) % 1000)  # Range
            ])
        except:
            estimates = [0.0] * 5
            
        return estimates
    
    def wieferich_test(self, n: int) -> float:
        """Test for Wieferich prime properties."""
        n = int(n)  # Convert to Python int
        if n <= 2:
            return 0.0
        try:
            return float(pow(2, n-1, n*n) == 1)
        except:
            return 0.0


class GNFSInspiredFeatures:
    """Features inspired by General Number Field Sieve characteristics."""
    
    def __init__(self):
        self.factor_bases = self._generate_factor_bases()
        self.polynomial_degrees = [3, 4, 5, 6]  # Common GNFS polynomial degrees
    
    def _generate_factor_bases(self) -> List[int]:
        """Generate small factor base for GNFS-inspired features."""
        return [p for p in range(2, 100) if isprime(p)]
    
    def extract_gnfs_features(self, n: int) -> List[float]:
        """Extract comprehensive GNFS-inspired features."""
        n = int(n)  # Convert to Python int
        features = []
        
        # Basic smoothness features
        features.append(self.smoothness_indicator(n, 50))
        features.append(self.smoothness_indicator(n, 100))
        features.append(self.smoothness_indicator(n, 200))
        
        # Polynomial selection features
        features.extend(self._murphy_e_estimates(n))
        features.extend(self._polynomial_properties(n))
        
        # Sieving-inspired features
        features.extend(self._sieve_density_estimates(n))
        features.extend(self.quadratic_sieve_features(n))
        
        # Number field features
        features.extend(self._number_field_properties(n))
        
        return features
    
    def _murphy_e_estimates(self, n: int) -> List[float]:
        """Estimate Murphy's E-score for polynomial quality."""
        estimates = []
        
        for degree in self.polynomial_degrees:
            try:
                # Simplified Murphy E-score estimate
                # Based on polynomial coefficient size and smoothness probability
                poly_bound = int(n**(1/degree))
                log_bound = np.log2(poly_bound) if poly_bound > 1 else 1
                murphy_estimate = 1.0 / (log_bound**degree)
                estimates.append(murphy_estimate)
            except:
                estimates.append(0.0)
                
        return estimates
    
    def _polynomial_properties(self, n: int) -> List[float]:
        """Extract properties related to polynomial selection."""
        properties = []
        
        try:
            # Root properties modulo small primes
            for p in self.factor_bases[:5]:
                if p > 2:
                    # Check if n has roots modulo p
                    has_root = any(pow(x, 2, p) == n % p for x in range(p))
                    properties.append(1.0 if has_root else 0.0)
                    
                    # Check cubic roots
                    has_cubic_root = any(pow(x, 3, p) == n % p for x in range(p))
                    properties.append(1.0 if has_cubic_root else 0.0)
            
            # Resultant estimates (simplified)
            sqrt_n = int(n**0.5)
            properties.append(float(gcd(sqrt_n, n)))
            properties.append(float(gcd(sqrt_n + 1, n)))
            
        except:
            properties = [0.0] * 12
            
        return properties
    
    def _sieve_density_estimates(self, n: int) -> List[float]:
        """Estimate sieving density and relation collection properties."""
        densities = []
        
        try:
            # Estimate density of smooth numbers near n
            log_n = np.log2(n) if n > 1 else 1
            
            for bound in [10, 20, 50]:
                # Simplified density estimate based on smooth number distribution
                density_est = 1.0 / (log_n * np.log2(bound) if bound > 1 else 1)
                densities.append(density_est)
            
            # Factor base size estimates
            optimal_fb_size = int(np.exp(0.5 * np.sqrt(log_n * np.log2(log_n))))
            densities.append(float(optimal_fb_size % 1000))
            
            # Sieving bound estimates
            sieve_bound = int(n**(1/3))
            densities.append(float(sieve_bound % 1000))
            
        except:
            densities = [0.0] * 5
            
        return densities
    
    def _number_field_properties(self, n: int) -> List[float]:
        """Extract algebraic number field properties."""
        properties = []
        
        try:
            # Class number heuristics (very simplified)
            for d in [-3, -4, -7, -8, -11]:  # Small discriminants
                class_estimate = float(abs(d)**0.5 * np.log2(abs(d)) if d != 0 else 1)
                properties.append(class_estimate % 100)
            
            # Regulator estimates
            log_n = np.log2(n) if n > 1 else 1
            reg_estimate = log_n**2  # Simplified regulator bound
            properties.append(reg_estimate % 1000)
            
        except:
            properties = [0.0] * 6
            
        return properties
    
    def smoothness_indicator(self, n: int, bound: int = 100) -> float:
        """
        Compute smoothness indicator - how 'smooth' the number is
        (related to how easily it factors with small primes).
        """
        temp = n
        smooth_part = 1
        
        for p in self.factor_bases:
            if p > bound:
                break
            while temp % p == 0:
                smooth_part *= p
                temp //= p
                
        return smooth_part / n if n > 0 else 0.0
    
    def quadratic_sieve_features(self, n: int) -> List[float]:
        """Extract features inspired by quadratic sieve method."""
        features = []
        
        # Check if n is a quadratic residue modulo small primes
        for p in self.factor_bases[:10]:
            if p == 2:
                features.append(1.0)  # 2 is always a QR mod 2
            else:
                qr = pow(n, (p-1)//2, p) if n % p != 0 else 0
                features.append(float(qr))
        
        # Continued fraction properties
        sqrt_n = int(n**0.5)
        features.append(float(sqrt_n * sqrt_n == n))  # Perfect square test
        features.append(float((sqrt_n + 1)**2 - n))   # Distance from next square
        
        return features


class FeatureEngineer:
    """Main feature engineering pipeline combining all methods."""
    
    def __init__(self):
        self.ecpp_extractor = ECPPFeatureExtractor()
        self.gnfs_extractor = GNFSInspiredFeatures()
    
    def extract_modular_residues(self, n: int, moduli: List[int] = [3, 5, 7, 11, 13]) -> List[float]:
        """Extract modular residues N mod m for specified moduli."""
        return [float(n % m) for m in moduli]
    
    def calculate_hamming_weight(self, n: int) -> float:
        """Calculate Hamming weight H(N) = sum of binary digits."""
        return float(bin(n).count('1'))
    
    def extract_all_features(self, n: int) -> np.ndarray:
        """Extract comprehensive feature vector for semiprime n."""
        n = int(n)  # Convert to Python int to avoid numpy issues
        features = []
        
        # Basic binary representation (30 bits for N < 1B)
        binary_repr = format(n, 'b')
        binary_features = [int(b) for b in binary_repr]
        while len(binary_features) < 30:
            binary_features.insert(0, 0)
        binary_features = binary_features[-30:]  # Take last 30 bits
        features.extend(binary_features)
        
        # UROP proposal features
        modular_residues = self.extract_modular_residues(n)
        hamming_weight = [self.calculate_hamming_weight(n)]
        features.extend(modular_residues)
        features.extend(hamming_weight)
        
        # ECPP-based features (enhanced)
        ecpp_features = self.ecpp_extractor.extract_ecpp_signature(n)
        features.extend(ecpp_features)
        
        # GNFS-inspired features (comprehensive)
        gnfs_features = self.gnfs_extractor.extract_gnfs_features(n)
        features.extend(gnfs_features)
        
        return np.array(features, dtype=np.float32)
    
    def prepare_training_data(self, semiprime_pairs: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from semiprime pairs.
        Returns (X, y) where X is features and y is smallest prime factor.
        """
        X = []
        y = []
        
        for p, q in semiprime_pairs:
            n = p * q
            smallest_prime = min(p, q)
            
            # Extract features for the semiprime
            features = self.extract_all_features(n)
            X.append(features)
            
            # Target is binary representation of smallest prime
            target_binary = format(smallest_prime, 'b')
            target_padded = [0] * (15 - len(target_binary)) + [int(b) for b in target_binary]
            y.append(target_padded)
        
        return np.array(X), np.array(y)


def generate_training_dataset(num_samples: int = 10000, max_bits: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Generate complete training dataset."""
    print(f"Generating {num_samples} semiprime samples with max {max_bits} bits...")
    
    # Initialize generators
    prime_gen = PrimeGenerator(max_bits)
    feature_eng = FeatureEngineer()
    
    # Generate prime pairs
    pairs = prime_gen.generate_prime_pairs(10, max_bits, num_samples)
    
    # Extract features and targets
    X, y = feature_eng.prepare_training_data(pairs)
    
    print(f"Dataset generated: X shape {X.shape}, y shape {y.shape}")
    return X, y


if __name__ == "__main__":
    # Test the feature extraction
    X, y = generate_training_dataset(1000, 20)
    print(f"Sample features shape: {X[0].shape}")
    print(f"Sample target shape: {y[0].shape}")
    print(f"First semiprime features: {X[0][:10]}...")
    print(f"First target (binary): {y[0]}")