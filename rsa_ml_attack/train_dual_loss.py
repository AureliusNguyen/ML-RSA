"""
Dual Loss Training
Implements dual loss for both p and q prediction as specified in methodology.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm


class DualOutputDataset(Dataset):
    """Dataset for dual p and q prediction."""
    
    def __init__(self, N_values, p_values, q_values, max_N_bits=20, max_factor_bits=10):
        self.X_binary = []
        self.y_p_binary = []
        self.y_q_binary = []
        
        # Calculate max bits needed for both p and q
        max_p = max(p_values)
        max_q = max(q_values)
        max_factor_bits = max(int(max_p).bit_length(), int(max_q).bit_length(), max_factor_bits)
        
        print(f"Using {max_factor_bits} bits for both p and q (max_p={max_p}, max_q={max_q})")
        
        for N, p, q in zip(N_values, p_values, q_values):
            # Input: N as binary
            N_binary = format(N, f'0{max_N_bits}b')
            self.X_binary.append([int(bit) for bit in N_binary])
            
            # Output 1: p as binary (use max_factor_bits for consistency)
            p_binary = format(p, f'0{max_factor_bits}b')
            self.y_p_binary.append([int(bit) for bit in p_binary])
            
            # Output 2: q as binary (same bit length as p)
            q_binary = format(q, f'0{max_factor_bits}b')
            self.y_q_binary.append([int(bit) for bit in q_binary])
        
        self.X = torch.FloatTensor(self.X_binary)
        self.y_p = torch.FloatTensor(self.y_p_binary)
        self.y_q = torch.FloatTensor(self.y_q_binary)
        self.output_bits = max_factor_bits
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_p[idx], self.y_q[idx]


class DualOutputLSTM(nn.Module):
    """
    LSTM with dual outputs for p and q prediction.
    Implements the dual loss 
    """
    
    def __init__(self, input_bits=20, output_bits=10, dropout_rate=0.3):
        super(DualOutputLSTM, self).__init__()
        
        # Shared LSTM backbone
        self.lstm1 = nn.LSTM(1, 128, batch_first=True)
        self.ln1 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lstm2 = nn.LSTM(128, 256, batch_first=True)
        self.ln2 = nn.LayerNorm(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.lstm3 = nn.LSTM(256, 512, batch_first=True)
        self.ln3 = nn.LayerNorm(512)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Shared dense layer
        self.shared_dense = nn.Linear(512, 256)
        self.shared_ln = nn.LayerNorm(256)
        self.shared_dropout = nn.Dropout(dropout_rate)
        
        # Separate heads for p and q prediction
        self.p_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_bits),
            nn.Sigmoid()
        )
        
        self.q_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_bits),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Shared backbone
        x = x.unsqueeze(2)  # (batch, bits, 1)
        
        x, _ = self.lstm1(x)
        x = self.ln1(x[:, -1, :])
        x = self.dropout1(x)
        
        x = x.unsqueeze(1)
        x, _ = self.lstm2(x)
        x = self.ln2(x[:, -1, :])
        x = self.dropout2(x)
        
        x = x.unsqueeze(1)
        x, _ = self.lstm3(x)
        x = self.ln3(x[:, -1, :])
        x = self.dropout3(x)
        
        # Shared feature extraction
        shared_features = torch.relu(self.shared_dense(x))
        shared_features = self.shared_ln(shared_features)
        shared_features = self.shared_dropout(shared_features)
        
        # Dual outputs
        p_pred = self.p_head(shared_features)
        q_pred = self.q_head(shared_features)
        
        return p_pred, q_pred


def dual_loss_function(p_pred, q_pred, p_true, q_true):
    """
    Dual loss function:
    L = -1/2k * Σ[y_i^(p) * log(ŷ_i^(p)) + y_i^(q) * log(ŷ_i^(q))]
    """
    bce = nn.BCELoss(reduction='sum')
    
    # Binary cross entropy for both outputs
    loss_p = bce(p_pred, p_true)
    loss_q = bce(q_pred, q_true)
    
    # Combined loss (as specified in proposal)
    total_loss = (loss_p + loss_q) / (2 * p_true.size(1))  # 2k normalization
    
    return total_loss, loss_p, loss_q


def calculate_dual_beta_metrics(p_pred, q_pred, p_true, q_true):
    """Calculate beta metrics for both p and q predictions."""
    # Convert to binary predictions
    p_binary = (p_pred > 0.5).float()
    q_binary = (q_pred > 0.5).float()
    
    # Calculate errors
    p_errors = (p_binary != p_true).sum(dim=1)
    q_errors = (q_binary != q_true).sum(dim=1)
    
    # Calculate beta metrics for p
    p_beta = {}
    for i in range(5):
        p_beta[f'p_beta_{i}'] = (p_errors <= i).float().mean().item()
    
    # Calculate beta metrics for q  
    q_beta = {}
    for i in range(5):
        q_beta[f'q_beta_{i}'] = (q_errors <= i).float().mean().item()
    
    # Combined beta metrics (both p AND q must be correct)
    combined_errors = p_errors + q_errors
    combined_beta = {}
    for i in range(5):
        # For combined metrics, we need both to be exactly right (stricter)
        combined_beta[f'combined_beta_{i}'] = ((p_errors <= i) & (q_errors <= i)).float().mean().item()
    
    return {**p_beta, **q_beta, **combined_beta}


def create_dual_dataset(csv_path, max_N_bits=20, max_p_bits=10):
    """Create dual output dataset from CSV with proper p,q calculation."""
    df = pd.read_csv(csv_path)
    
    N_values = df['N'].values
    p_values = df['p'].values
    q_values = []
    
    # Calculate q = N / p for each sample
    for N, p in zip(N_values, p_values):
        q = N // p
        q_values.append(q)
    
    return DualOutputDataset(N_values, p_values, q_values, max_N_bits, max_p_bits)


def train_dual_model(train_loader, test_loader, device, epochs=120):
    """Train dual output model with dual loss function."""
    
    # Get dimensions from data
    sample_x, sample_p, sample_q = next(iter(train_loader))
    input_bits = sample_x.size(1)
    output_bits = sample_p.size(1)
    
    # Verify both outputs have same dimensions
    assert sample_p.size(1) == sample_q.size(1), f"p and q must have same bit length: {sample_p.size(1)} vs {sample_q.size(1)}"
    
    print(f"Training Dual LSTM: {input_bits}-bit input → {output_bits}-bit output (both p and q)")
    
    # Initialize model
    model = DualOutputLSTM(input_bits, output_bits).to(device)
    
    # RMSprop optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    training_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        total_p_loss = 0
        total_q_loss = 0
        
        for batch_x, batch_p, batch_q in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_x = batch_x.to(device)
            batch_p, batch_q = batch_p.to(device), batch_q.to(device)
            
            optimizer.zero_grad()
            p_pred, q_pred = model(batch_x)
            
            # Dual loss function
            loss, p_loss, q_loss = dual_loss_function(p_pred, q_pred, batch_p, batch_q)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_p_loss += p_loss.item()
            total_q_loss += q_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_p_loss = total_p_loss / len(train_loader)
        avg_q_loss = total_q_loss / len(train_loader)
        
        # Evaluate every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            model.eval()
            test_loss = 0
            all_p_predictions = []
            all_q_predictions = []
            all_p_targets = []
            all_q_targets = []
            
            with torch.no_grad():
                for batch_x, batch_p, batch_q in test_loader:
                    batch_x = batch_x.to(device)
                    batch_p, batch_q = batch_p.to(device), batch_q.to(device)
                    
                    p_pred, q_pred = model(batch_x)
                    loss, _, _ = dual_loss_function(p_pred, q_pred, batch_p, batch_q)
                    test_loss += loss.item()
                    
                    all_p_predictions.append(p_pred)
                    all_q_predictions.append(q_pred)
                    all_p_targets.append(batch_p)
                    all_q_targets.append(batch_q)
            
            all_p_pred = torch.cat(all_p_predictions, dim=0)
            all_q_pred = torch.cat(all_q_predictions, dim=0)
            all_p_true = torch.cat(all_p_targets, dim=0)
            all_q_true = torch.cat(all_q_targets, dim=0)
            
            # Calculate dual beta metrics
            dual_metrics = calculate_dual_beta_metrics(all_p_pred, all_q_pred, all_p_true, all_q_true)
            avg_test_loss = test_loss / len(test_loader)
            
            print(f"Epoch {epoch+1:3d}: Total Loss={avg_loss:.4f}, Test Loss={avg_test_loss:.4f}")
            print(f"         p_beta_0={dual_metrics['p_beta_0']:.4f}, q_beta_0={dual_metrics['q_beta_0']:.4f}, "
                  f"Combined_beta_0={dual_metrics['combined_beta_0']:.4f}")
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'train_p_loss': avg_p_loss,
                'train_q_loss': avg_q_loss,
                'test_loss': avg_test_loss,
                **dual_metrics
            })
            
            model.train()
    
    # Final evaluation
    model.eval()
    final_p_pred = []
    final_q_pred = []
    final_p_true = []
    final_q_true = []
    
    with torch.no_grad():
        for batch_x, batch_p, batch_q in test_loader:
            batch_x = batch_x.to(device)
            batch_p, batch_q = batch_p.to(device), batch_q.to(device)
            
            p_pred, q_pred = model(batch_x)
            final_p_pred.append(p_pred)
            final_q_pred.append(q_pred)
            final_p_true.append(batch_p)
            final_q_true.append(batch_q)
    
    final_p_pred = torch.cat(final_p_pred, dim=0)
    final_q_pred = torch.cat(final_q_pred, dim=0)
    final_p_true = torch.cat(final_p_true, dim=0)
    final_q_true = torch.cat(final_q_true, dim=0)
    
    final_dual_metrics = calculate_dual_beta_metrics(final_p_pred, final_q_pred, final_p_true, final_q_true)
    
    print(f"\nFinal Dual Results:")
    print(f"p prediction - beta_0: {final_dual_metrics['p_beta_0']:.4f}, beta_1: {final_dual_metrics['p_beta_1']:.4f}")
    print(f"q prediction - beta_0: {final_dual_metrics['q_beta_0']:.4f}, beta_1: {final_dual_metrics['q_beta_1']:.4f}")
    print(f"Combined (both correct) - beta_0: {final_dual_metrics['combined_beta_0']:.4f}")
    
    return model, training_history, final_dual_metrics


def main():
    parser = argparse.ArgumentParser(description='Dual Loss Training')
    parser.add_argument('--dataset', type=str, default='small', 
                       choices=['tiny', 'small', 'medium', 'large'],
                       help='Dataset scale to use')
    parser.add_argument('--epochs', type=int, default=120, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    
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
    
    # Determine bit sizes
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    max_N = max(train_df['N'].max(), test_df['N'].max())
    max_p = max(train_df['p'].max(), test_df['p'].max())
    
    # Calculate max q across both datasets for proper bit sizing
    train_q_values = (train_df['N'] // train_df['p']).values
    test_q_values = (test_df['N'] // test_df['p']).values  
    max_q = max(max(train_q_values), max(test_q_values))
    
    max_N_bits = int(max_N).bit_length()
    # Use max of p and q bit requirements for consistent sizing
    max_factor_bits = max(int(max_p).bit_length(), int(max_q).bit_length())
    
    print(f"Using {max_N_bits} bits for N, {max_factor_bits} bits for p and q (max_p={max_p}, max_q={max_q})")
    
    # Create dual datasets with 80-10-10 split 
    train_dataset = create_dual_dataset(train_path, max_N_bits, max_factor_bits)
    
    test_dataset = create_dual_dataset(test_path, max_N_bits, max_factor_bits)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Train model with dual loss
    model, history, final_metrics = train_dual_model(
        train_loader, test_loader, device, epochs=args.epochs)
    
    # Save results
    results_dir = f"experiments/dual_training_{args.dataset}"
    os.makedirs(results_dir, exist_ok=True)
    
    torch.save(model.state_dict(), f"{results_dir}/dual_lstm_model.pth")
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(f"{results_dir}/dual_training_history.csv", index=False)
    
    import json
    with open(f"{results_dir}/dual_final_metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"Dual loss results saved to: {results_dir}")


if __name__ == "__main__":
    main()