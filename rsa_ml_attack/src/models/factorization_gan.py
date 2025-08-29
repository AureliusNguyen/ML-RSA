"""
Generative Adversarial Network for RSA semiprime factorization.
Generator learns to produce prime factors, discriminator validates primality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import random


class PrimeFactorGenerator(nn.Module):
    """
    Generator network that learns to produce prime factors given a semiprime.
    Uses mathematical insights to guide factor generation.
    """
    
    def __init__(self, 
                 input_size: int = 120,  # Enhanced feature size
                 latent_dim: int = 128,
                 output_size: int = 15,
                 hidden_dims: list = [256, 512, 256]):
        super(PrimeFactorGenerator, self).__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.output_size = output_size
        
        # Input processing layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size + latent_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.LeakyReLU(0.2)
        )
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.LayerNorm(hidden_dims[i+1]),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                )
            )
        
        # Mathematical constraint layer
        self.constraint_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.LayerNorm(hidden_dims[-1] // 2),
            nn.LeakyReLU(0.2)
        )
        
        # Factor generation layers
        self.factor_head = nn.Sequential(
            nn.Linear(hidden_dims[-1] // 2, output_size * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(output_size * 2, output_size),
            nn.Sigmoid()  # Binary output for prime factor bits
        )
        
        # Primality enhancement layer
        self.primality_layer = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, semiprime_features: torch.Tensor, noise: Optional[torch.Tensor] = None):
        batch_size = semiprime_features.size(0)
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(batch_size, self.latent_dim, device=semiprime_features.device)
        
        # Concatenate input features with noise
        x = torch.cat([semiprime_features, noise], dim=1)
        
        # Forward pass through generator
        x = self.input_layer(x)
        
        for layer in self.hidden_layers:
            residual = x
            x = layer(x)
            # Add residual connection if dimensions match
            if residual.size(-1) == x.size(-1):
                x = x + residual
        
        # Apply mathematical constraints
        x = self.constraint_layer(x)
        
        # Generate factor bits
        factor_bits = self.factor_head(x)
        
        # Enhance primality properties
        factor_bits = self.primality_layer(factor_bits)
        
        # Enforce odd number constraint (last bit = 1) - non-inplace version
        last_bit = torch.sigmoid(factor_bits[:, -1:] + 2.0)  # Bias toward 1
        factor_bits = torch.cat([factor_bits[:, :-1], last_bit], dim=1)
        
        return factor_bits


class PrimeFactorDiscriminator(nn.Module):
    """
    Discriminator network that validates whether generated factors are:
    1. Actually prime
    2. Valid factors of the input semiprime
    3. Mathematically consistent
    """
    
    def __init__(self, 
                 semiprime_size: int = 120,
                 factor_size: int = 15,
                 hidden_dims: list = [256, 512, 256, 128]):
        super(PrimeFactorDiscriminator, self).__init__()
        
        input_size = semiprime_size + factor_size
        
        # Input processing with mathematical feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # Hidden layers for pattern recognition
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.LayerNorm(hidden_dims[i+1]),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                )
            )
        
        # Multi-task outputs
        self.primality_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.validity_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.authenticity_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, semiprime_features: torch.Tensor, factor_bits: torch.Tensor):
        # Concatenate semiprime and factor features
        x = torch.cat([semiprime_features, factor_bits], dim=1)
        
        # Extract features
        x = self.feature_extractor(x)
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Multi-task predictions
        primality_score = self.primality_head(x)    # Is the factor prime?
        validity_score = self.validity_head(x)      # Is it a valid factor?
        authenticity_score = self.authenticity_head(x)  # Is it real vs generated?
        
        return {
            'primality': primality_score,
            'validity': validity_score,
            'authenticity': authenticity_score
        }


class FactorizationGAN:
    """
    Complete GAN system for RSA semiprime factorization.
    Combines generator and discriminator with specialized training procedures.
    """
    
    def __init__(self, 
                 input_size: int = 120,
                 factor_size: int = 15,
                 latent_dim: int = 128,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.input_size = input_size
        self.factor_size = factor_size
        
        # Initialize networks
        self.generator = PrimeFactorGenerator(
            input_size=input_size,
            latent_dim=latent_dim,
            output_size=factor_size
        ).to(device)
        
        self.discriminator = PrimeFactorDiscriminator(
            semiprime_size=input_size,
            factor_size=factor_size
        ).to(device)
        
        # Optimizers with different learning rates
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), 
            lr=0.0001, 
            betas=(0.5, 0.999)
        )
        
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=0.0004, 
            betas=(0.5, 0.999)
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.factor_loss = nn.MSELoss()
        
        # Training statistics
        self.g_losses = []
        self.d_losses = []
    
    def _bits_to_number(self, bits: torch.Tensor) -> torch.Tensor:
        """Convert binary representation to integer."""
        powers = torch.pow(2, torch.arange(bits.size(1)-1, -1, -1, dtype=torch.float32, device=bits.device))
        return torch.sum(bits * powers, dim=1)
    
    def _check_factorization(self, semiprime_bits: torch.Tensor, factor_bits: torch.Tensor) -> torch.Tensor:
        """
        Check if generated factor actually divides the semiprime.
        Returns validity scores (1.0 for valid factors, 0.0 for invalid).
        """
        batch_size = semiprime_bits.size(0)
        validity_scores = torch.zeros(batch_size, 1, device=self.device)
        
        # Convert bits to numbers (simplified for small examples)
        semiprimes = self._bits_to_number(semiprime_bits[:, :30])  # Use first 30 bits
        factors = self._bits_to_number(factor_bits)
        
        # Check divisibility (avoid division by zero)
        valid_factors = factors > 1
        
        for i in range(batch_size):
            if valid_factors[i] and semiprimes[i] > 1:
                remainder = semiprimes[i] % factors[i]
                validity_scores[i] = 1.0 if remainder < 0.01 else 0.0  # Allow small numerical errors
        
        return validity_scores
    
    def train_discriminator(self, real_semiprimes: torch.Tensor, real_factors: torch.Tensor):
        """Train discriminator to distinguish real vs generated factors."""
        batch_size = real_semiprimes.size(0)
        
        # Train with real data
        real_labels = torch.ones(batch_size, 1, device=self.device)
        real_scores = self.discriminator(real_semiprimes, real_factors)
        
        real_loss = (
            self.adversarial_loss(real_scores['authenticity'], real_labels) +
            self.adversarial_loss(real_scores['primality'], real_labels) +
            self.adversarial_loss(real_scores['validity'], real_labels)
        )
        
        # Train with fake data
        noise = torch.randn(batch_size, 128, device=self.device)
        fake_factors = self.generator(real_semiprimes, noise)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        fake_scores = self.discriminator(real_semiprimes, fake_factors.detach())
        
        # Check actual validity for fake factors
        actual_validity = self._check_factorization(real_semiprimes, fake_factors.detach())
        
        fake_loss = (
            self.adversarial_loss(fake_scores['authenticity'], fake_labels) +
            self.adversarial_loss(fake_scores['primality'], fake_labels) +
            self.adversarial_loss(fake_scores['validity'], actual_validity)  # Use actual validity
        )
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        
        # Update discriminator
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
    
    def train_generator(self, real_semiprimes: torch.Tensor, real_factors: torch.Tensor):
        """Train generator to fool discriminator and produce valid factors."""
        batch_size = real_semiprimes.size(0)
        
        # Generate fake factors
        noise = torch.randn(batch_size, 128, device=self.device)
        generated_factors = self.generator(real_semiprimes, noise)
        
        # Discriminator's opinion on generated factors
        fake_scores = self.discriminator(real_semiprimes, generated_factors)
        
        # Generator wants discriminator to think factors are real and valid
        real_labels = torch.ones(batch_size, 1, device=self.device)
        
        adversarial_loss = (
            self.adversarial_loss(fake_scores['authenticity'], real_labels) +
            self.adversarial_loss(fake_scores['primality'], real_labels) +
            self.adversarial_loss(fake_scores['validity'], real_labels)
        )
        
        # Mathematical validity loss
        validity_scores = self._check_factorization(real_semiprimes, generated_factors)
        validity_loss = self.factor_loss(fake_scores['validity'], validity_scores)
        
        # Factor reconstruction loss (encourage generating actual factors)
        reconstruction_loss = self.factor_loss(generated_factors, real_factors)
        
        # Total generator loss
        g_loss = adversarial_loss + 0.5 * validity_loss + 0.1 * reconstruction_loss
        
        # Update generator
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
    
    def train_epoch(self, dataloader):
        """Train both networks for one epoch."""
        epoch_g_losses = []
        epoch_d_losses = []
        
        for batch_features, batch_factors in dataloader:
            batch_features = batch_features.to(self.device)
            batch_factors = batch_factors.to(self.device)
            
            # Train discriminator
            d_loss = self.train_discriminator(batch_features, batch_factors)
            epoch_d_losses.append(d_loss)
            
            # Train generator (every other iteration to balance training)
            if len(epoch_d_losses) % 2 == 0:
                g_loss = self.train_generator(batch_features, batch_factors)
                epoch_g_losses.append(g_loss)
        
        avg_g_loss = np.mean(epoch_g_losses) if epoch_g_losses else 0
        avg_d_loss = np.mean(epoch_d_losses)
        
        self.g_losses.append(avg_g_loss)
        self.d_losses.append(avg_d_loss)
        
        return avg_g_loss, avg_d_loss
    
    def generate_factors(self, semiprime_features: torch.Tensor, num_samples: int = 1):
        """Generate potential prime factors for given semiprimes."""
        self.generator.eval()
        
        batch_size = semiprime_features.size(0)
        all_factors = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                noise = torch.randn(batch_size, 128, device=self.device)
                factors = self.generator(semiprime_features, noise)
                all_factors.append(factors)
        
        return torch.stack(all_factors, dim=1)  # [batch, num_samples, factor_bits]
    
    def evaluate_factorization_accuracy(self, dataloader):
        """Evaluate the GAN's factorization accuracy."""
        self.generator.eval()
        
        correct_factors = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_features, batch_true_factors in dataloader:
                batch_features = batch_features.to(self.device)
                batch_true_factors = batch_true_factors.to(self.device)
                
                # Generate multiple factor candidates
                generated_factors = self.generate_factors(batch_features, num_samples=5)
                
                # Check which generated factors match true factors
                for i in range(generated_factors.size(0)):
                    true_factor = batch_true_factors[i]
                    candidates = generated_factors[i]  # [num_samples, factor_bits]
                    
                    # Check if any candidate matches the true factor
                    matches = torch.all(torch.abs(candidates - true_factor) < 0.5, dim=1)
                    if torch.any(matches):
                        correct_factors += 1
                    
                    total_samples += 1
        
        accuracy = correct_factors / total_samples if total_samples > 0 else 0
        return accuracy


def create_gan_dataloader(X, y, batch_size=32):
    """Create dataloader for GAN training."""
    from torch.utils.data import Dataset, DataLoader
    
    class GANDataset(Dataset):
        def __init__(self, X, y):
            self.features = torch.FloatTensor(X)
            self.factors = torch.FloatTensor(y)
        
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, idx):
            return self.features[idx], self.factors[idx]
    
    dataset = GANDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    # Test GAN architecture
    gan = FactorizationGAN(input_size=120, factor_size=15)
    
    print(f"Generator parameters: {sum(p.numel() for p in gan.generator.parameters())}")
    print(f"Discriminator parameters: {sum(p.numel() for p in gan.discriminator.parameters())}")
    
    # Test forward pass
    test_features = torch.randn(16, 120)
    test_factors = torch.randn(16, 15)
    
    with torch.no_grad():
        generated = gan.generator(test_features)
        discriminated = gan.discriminator(test_features, generated)
        
        print(f"Generated factors shape: {generated.shape}")
        print(f"Discriminator outputs: {list(discriminated.keys())}")
        print(f"Sample discriminator scores: {discriminated['authenticity'][:3].flatten()}")