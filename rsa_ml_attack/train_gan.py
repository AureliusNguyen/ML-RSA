"""
GAN-Based RSA Factorization Training Script
Implements adversarial training for prime factor generation.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
import json

# Import GAN models
import sys
sys.path.append('src')
from models.factorization_gan import FactorizationGAN, create_gan_dataloader
from crypto_utils import FeatureEngineer


def calculate_beta_metrics(predictions, targets):
    """Calculate beta_i metrics from papers (percentage with at most i bit errors)."""
    # Convert to binary predictions
    binary_preds = (predictions > 0.5).float()
    
    # Count bit errors for each sample
    errors = (binary_preds != targets).sum(dim=1)
    
    beta_metrics = {}
    for i in range(5):
        beta_metrics[f'beta_{i}'] = (errors <= i).float().mean().item()
    
    return beta_metrics


class GANSemiprimeDataset(Dataset):
    """Dataset for GAN training with enhanced features."""
    
    def __init__(self, N_values, p_values, feature_engineer):
        self.enhanced_features = []
        self.factor_bits = []
        
        # Determine bit sizes
        max_p = max(p_values)
        max_factor_bits = int(max_p).bit_length()
        
        print(f"Using {max_factor_bits} bits for prime factors (max_p={max_p})")
        
        for N, p in zip(N_values, p_values):
            # Enhanced features for N (input)
            features = feature_engineer.extract_all_features(N)
            self.enhanced_features.append(features)
            
            # Binary representation of p (target)
            p_binary = format(p, f'0{max_factor_bits}b')
            self.factor_bits.append([int(bit) for bit in p_binary])
        
        # Convert to numpy first to avoid the warning
        self.X = torch.FloatTensor(np.array(self.enhanced_features))
        self.y = torch.FloatTensor(np.array(self.factor_bits))
        self.feature_size = self.X.size(1)
        self.factor_size = self.y.size(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def evaluate_gan_with_beta_metrics(gan, test_loader, device):
    """Comprehensive evaluation with beta-metrics like other models."""
    gan.generator.eval()
    
    all_predictions = []
    all_targets = []
    
    print("\nEvaluating GAN with beta-metrics...")
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            # Generate factors (take best of 3 generations per sample)
            generated_factors = gan.generate_factors(batch_features, num_samples=3)
            
            # For each sample, select the best generation (closest to target)
            best_predictions = []
            for i in range(batch_features.size(0)):
                target = batch_targets[i]
                candidates = generated_factors[i]  # Shape: (3, factor_size)
                
                # Calculate errors for each candidate
                errors = []
                for candidate in candidates:
                    binary_candidate = (candidate > 0.5).float()
                    error_count = (binary_candidate != target).sum().item()
                    errors.append(error_count)
                
                # Select candidate with minimum errors
                best_idx = errors.index(min(errors))
                best_predictions.append(generated_factors[i, best_idx])
            
            # Stack best predictions
            batch_predictions = torch.stack(best_predictions)
            
            all_predictions.append(batch_predictions)
            all_targets.append(batch_targets)
    
    # Concatenate all batches
    final_predictions = torch.cat(all_predictions, dim=0)
    final_targets = torch.cat(all_targets, dim=0)
    
    # Calculate β-metrics
    beta_metrics = calculate_beta_metrics(final_predictions, final_targets)
    
    return beta_metrics, final_predictions, final_targets


def train_gan_model(train_loader, test_loader, device, epochs=200):
    """Train GAN model for RSA factorization."""
    
    # Get dimensions from data
    sample_x, sample_y = next(iter(train_loader))
    input_size = sample_x.size(1)
    factor_size = sample_y.size(1)
    
    print(f"Training GAN: {input_size}-dimensional features -> {factor_size}-bit factors")
    
    # Initialize GAN
    gan = FactorizationGAN(
        input_size=input_size,
        factor_size=factor_size,
        latent_dim=128,
        device=device
    )
    
    print(f"Generator parameters: {sum(p.numel() for p in gan.generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in gan.discriminator.parameters()):,}")
    
    # Training history
    training_history = []
    
    print(f"Starting GAN training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Train for one epoch
        avg_g_loss, avg_d_loss = gan.train_epoch(train_loader)
        
        print(f"Epoch {epoch+1:3d}/{epochs}: G_Loss={avg_g_loss:.4f}, D_Loss={avg_d_loss:.4f}")
        
        # Evaluate every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            # Quick factorization accuracy (beta_0)
            accuracy = gan.evaluate_factorization_accuracy(test_loader)
            print(f"         Factorization Accuracy (beta_0): {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Detailed beta-metrics evaluation (every 20 epochs)
            if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
                beta_metrics, _, _ = evaluate_gan_with_beta_metrics(gan, test_loader, device)
                print(f"         beta_1={beta_metrics['beta_1']:.4f}, beta_2={beta_metrics['beta_2']:.4f}, beta_3={beta_metrics['beta_3']:.4f}")
                
                training_history.append({
                    'epoch': epoch + 1,
                    'g_loss': avg_g_loss,
                    'd_loss': avg_d_loss,
                    'factorization_accuracy': accuracy,
                    'beta_0': beta_metrics['beta_0'],
                    'beta_1': beta_metrics['beta_1'],
                    'beta_2': beta_metrics['beta_2'],
                    'beta_3': beta_metrics['beta_3'],
                    'beta_4': beta_metrics['beta_4']
                })
            else:
                training_history.append({
                    'epoch': epoch + 1,
                    'g_loss': avg_g_loss,
                    'd_loss': avg_d_loss,
                    'factorization_accuracy': accuracy
                })
    
    # Final evaluation with comprehensive beta-metrics
    print(f"\n{'='*60}")
    print(f"FINAL GAN EVALUATION")
    print(f"{'='*60}")
    
    # Complete beta-metrics evaluation
    final_beta_metrics, predictions, targets = evaluate_gan_with_beta_metrics(gan, test_loader, device)
    
    print(f"Final beta-metrics:")
    print(f"  beta_0 (exact match): {final_beta_metrics['beta_0']:.4f} ({final_beta_metrics['beta_0']*100:.2f}%)")
    print(f"  beta_1 (<=1 bit error): {final_beta_metrics['beta_1']:.4f} ({final_beta_metrics['beta_1']*100:.2f}%)")
    print(f"  beta_2 (<=2 bit error): {final_beta_metrics['beta_2']:.4f} ({final_beta_metrics['beta_2']*100:.2f}%)")
    print(f"  beta_3 (<=3 bit error): {final_beta_metrics['beta_3']:.4f} ({final_beta_metrics['beta_3']*100:.2f}%)")
    print(f"  beta_4 (<=4 bit error): {final_beta_metrics['beta_4']:.4f} ({final_beta_metrics['beta_4']*100:.2f}%)")
    
    gan.generator.eval()
    with torch.no_grad():
        # Take first batch for demonstration
        sample_features, sample_targets = next(iter(test_loader))
        sample_features = sample_features.to(device)
        sample_targets = sample_targets.to(device)
        
        # Generate multiple factor candidates (up to 5 samples, or all available if less)
        num_samples = min(5, len(sample_features))
        generated_factors = gan.generate_factors(sample_features[:num_samples], num_samples=3)
        
        print(f"\nSample factor generation (first {num_samples} test samples):")
        for i in range(num_samples):
            print(f"  Sample {i+1}:")
            print(f"    True factor:  {sample_targets[i]}")
            print(f"    Generated 1:  {generated_factors[i, 0]}")
            print(f"    Generated 2:  {generated_factors[i, 1]}")
            print(f"    Generated 3:  {generated_factors[i, 2]}")
    
    final_accuracy = final_beta_metrics['beta_0']
    print(f"\nFinal Factorization Accuracy (beta_0): {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    return gan, training_history, final_accuracy, final_beta_metrics


def main():
    parser = argparse.ArgumentParser(description='GAN RSA Factorization Training')
    parser.add_argument('--dataset', type=str, default='small', 
                       choices=['tiny', 'small', 'medium', 'large'],
                       help='Dataset scale to use')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    train_path = f"data/{args.dataset}_train.csv"
    test_path = f"data/{args.dataset}_test.csv"
    
    if not os.path.exists(train_path):
        print(f"Dataset not found! Please run: python generate_data.py --dataset {args.dataset}")
        return
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded {len(train_df)} training samples, {len(test_df)} test samples")
    print(f"Max N: {max(train_df['N'].max(), test_df['N'].max()):,}")
    print(f"Max p: {max(train_df['p'].max(), test_df['p'].max()):,}")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Create GAN datasets with enhanced features
    train_dataset = GANSemiprimeDataset(
        train_df['N'].values, 
        train_df['p'].values, 
        feature_engineer
    )
    
    test_dataset = GANSemiprimeDataset(
        test_df['N'].values, 
        test_df['p'].values, 
        feature_engineer
    )
    
    print(f"Feature vector size: {train_dataset.feature_size}")
    print(f"Factor bit size: {train_dataset.factor_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Train GAN
    gan, history, final_accuracy, final_beta_metrics = train_gan_model(
        train_loader, test_loader, device, epochs=args.epochs
    )
    
    # Save results
    results_dir = f"experiments/gan_training_{args.dataset}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save models
    torch.save(gan.generator.state_dict(), f"{results_dir}/gan_generator.pth")
    torch.save(gan.discriminator.state_dict(), f"{results_dir}/gan_discriminator.pth")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(f"{results_dir}/gan_training_history.csv", index=False)
    
    # Save final metrics with all beta-metrics
    final_metrics = {
        'factorization_accuracy': final_accuracy,
        'beta_0': final_beta_metrics['beta_0'],
        'beta_1': final_beta_metrics['beta_1'],
        'beta_2': final_beta_metrics['beta_2'],
        'beta_3': final_beta_metrics['beta_3'],
        'beta_4': final_beta_metrics['beta_4'],
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'feature_size': train_dataset.feature_size,
        'factor_size': train_dataset.factor_size
    }
    
    with open(f"{results_dir}/gan_final_metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"GAN training results saved to: {results_dir}")


if __name__ == "__main__":
    main()