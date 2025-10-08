"""
Enhanced Feature Models Training Script
Trains Transformer and Hybrid CNN-RNN models with comprehensive features.
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

# Import enhanced models
import sys
sys.path.append('src')
from models.transformer_factorizer import TransformerFactorizer
from models.hybrid_cnn_rnn import HybridCNNRNN
from crypto_utils import FeatureEngineer


class EnhancedSemiprimeDataset(Dataset):
    """Dataset for enhanced feature models."""
    
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
        
        # Convert to numpy arrays first for better performance
        self.X = torch.FloatTensor(np.array(self.enhanced_features))
        self.y = torch.FloatTensor(np.array(self.factor_bits))
        self.feature_size = self.X.size(1)
        self.factor_size = self.y.size(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def calculate_beta_metrics(predictions, targets):
    """Calculate beta_i metrics from papers."""
    binary_preds = (predictions > 0.5).float()
    errors = (binary_preds != targets).sum(dim=1)
    
    beta_metrics = {}
    for i in range(5):
        beta_metrics[f'beta_{i}'] = (errors <= i).float().mean().item()
    
    return beta_metrics


def train_enhanced_model(model_type, train_loader, test_loader, device, epochs=120):
    """Train enhanced feature models."""
    
    # Get dimensions from data
    sample_x, sample_y = next(iter(train_loader))
    feature_size = sample_x.size(1)
    factor_size = sample_y.size(1)
    
    print(f"Training {model_type}: {feature_size}-dimensional features -> {factor_size}-bit factors")
    
    # Initialize model
    if model_type == "transformer":
        # TransformerFactorizer expects a dictionary of input sizes
        # For now, we'll treat all features as one type
        input_sizes = {'combined': feature_size}
        model = TransformerFactorizer(
            input_sizes=input_sizes,
            d_model=256,
            num_heads=8,
            num_layers=4,
            output_size=factor_size
        ).to(device)
    elif model_type == "hybrid":
        model = HybridCNNRNN(
            input_size=feature_size,
            output_size=factor_size
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    model.train()
    training_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{epochs}")
        
        for batch_x, batch_y in progress_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            # Handle transformer combined mode
            if hasattr(model, 'feature_embedding') and hasattr(model.feature_embedding, 'mode'):
                outputs = model(combined_features=batch_x)
            else:
                outputs = model(batch_x)
            
            # Handle transformer returning (predictions, attention_weights)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take just the predictions
                
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            test_loss = 0
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    # Handle transformer combined mode
                    if hasattr(model, 'feature_embedding') and hasattr(model.feature_embedding, 'mode'):
                        outputs = model(combined_features=batch_x)
                    else:
                        outputs = model(batch_x)
                    
                    # Handle transformer returning (predictions, attention_weights)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Take just the predictions
                        
                    test_loss += criterion(outputs, batch_y).item()
                    
                    all_predictions.append(outputs)
                    all_targets.append(batch_y)
            
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Calculate beta metrics
            beta_metrics = calculate_beta_metrics(all_predictions, all_targets)
            avg_test_loss = test_loss / len(test_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1:3d}: Train Loss={avg_loss:.4f}, Test Loss={avg_test_loss:.4f}, LR={current_lr:.6f}")
            print(f"         beta_0={beta_metrics['beta_0']:.4f}, beta_1={beta_metrics['beta_1']:.4f}, "
                  f"beta_2={beta_metrics['beta_2']:.4f}, beta_3={beta_metrics['beta_3']:.4f}")
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'test_loss': avg_test_loss,
                'learning_rate': current_lr,
                **beta_metrics
            })
            
            model.train()
    
    # Final evaluation
    model.eval()
    final_predictions = []
    final_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Handle transformer combined mode
            if hasattr(model, 'feature_embedding') and hasattr(model.feature_embedding, 'mode'):
                outputs = model(combined_features=batch_x)
            else:
                outputs = model(batch_x)
            
            # Handle transformer returning (predictions, attention_weights)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take just the predictions
                
            final_predictions.append(outputs)
            final_targets.append(batch_y)
    
    final_predictions = torch.cat(final_predictions, dim=0)
    final_targets = torch.cat(final_targets, dim=0)
    final_beta_metrics = calculate_beta_metrics(final_predictions, final_targets)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - {model_type.upper()} WITH ENHANCED FEATURES")
    print(f"{'='*60}")
    print(f"beta_0 (Exact Match):   {final_beta_metrics['beta_0']:.4f} ({final_beta_metrics['beta_0']*100:.2f}%)")
    print(f"beta_1 (≤1 bit error):  {final_beta_metrics['beta_1']:.4f} ({final_beta_metrics['beta_1']*100:.2f}%)")
    print(f"beta_2 (≤2 bit error):  {final_beta_metrics['beta_2']:.4f} ({final_beta_metrics['beta_2']*100:.2f}%)")
    print(f"beta_3 (≤3 bit error):  {final_beta_metrics['beta_3']:.4f} ({final_beta_metrics['beta_3']*100:.2f}%)")
    print(f"beta_4 (≤4 bit error):  {final_beta_metrics['beta_4']:.4f} ({final_beta_metrics['beta_4']*100:.2f}%)")
    
    return model, training_history, final_beta_metrics


def main():
    parser = argparse.ArgumentParser(description='Enhanced Feature Models Training')
    parser.add_argument('--model', type=str, default='transformer', 
                       choices=['transformer', 'hybrid', 'both'],
                       help='Model type to train')
    parser.add_argument('--dataset', type=str, default='small', 
                       choices=['tiny', 'small', 'medium', 'large'],
                       help='Dataset scale to use')
    parser.add_argument('--epochs', type=int, default=120, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
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
    print("Initializing feature engineer...")
    feature_engineer = FeatureEngineer()
    
    # Create enhanced datasets
    print("Creating enhanced feature datasets...")
    train_dataset = EnhancedSemiprimeDataset(
        train_df['N'].values, 
        train_df['p'].values, 
        feature_engineer
    )
    
    test_dataset = EnhancedSemiprimeDataset(
        test_df['N'].values, 
        test_df['p'].values, 
        feature_engineer
    )
    
    print(f"Feature vector size: {train_dataset.feature_size}")
    print(f"Factor bit size: {train_dataset.factor_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Train models
    models_to_train = ['transformer', 'hybrid'] if args.model == 'both' else [args.model]
    
    for model_type in models_to_train:
        print(f"\n{'='*80}")
        print(f"TRAINING {model_type.upper()} MODEL")
        print(f"{'='*80}")
        
        model, history, final_metrics = train_enhanced_model(
            model_type, train_loader, test_loader, device, epochs=args.epochs
        )
        
        # Save results
        results_dir = f"experiments/{model_type}_enhanced_{args.dataset}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), f"{results_dir}/{model_type}_enhanced_model.pth")
        
        # Save training history
        history_df = pd.DataFrame(history)
        history_df.to_csv(f"{results_dir}/{model_type}_training_history.csv", index=False)
        
        # Save final metrics
        final_metrics['model_type'] = model_type
        final_metrics['dataset'] = args.dataset
        final_metrics['epochs'] = args.epochs
        final_metrics['batch_size'] = args.batch_size
        final_metrics['feature_size'] = train_dataset.feature_size
        final_metrics['factor_size'] = train_dataset.factor_size
        
        with open(f"{results_dir}/{model_type}_final_metrics.json", 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"{model_type.capitalize()} results saved to: {results_dir}")


if __name__ == "__main__":
    main()