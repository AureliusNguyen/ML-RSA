"""
Kaggle/Colab Compatible Binary Framework Training Script
Fixed to work with Jupyter notebook environments
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import json

# Binary conversion utilities following papers exactly
def convert_to_binary_vectors(N_values, p_values, max_N_bits=20, max_p_bits=10):
    """Convert integers to binary vectors as per Murat et al. methodology."""
    X_binary = []
    y_binary = []
    
    for N, p in zip(N_values, p_values):
        # Convert N to binary (input)
        N_binary = format(N, f'0{max_N_bits}b')
        X_binary.append([int(bit) for bit in N_binary])
        
        # Convert p to binary (output)
        p_binary = format(p, f'0{max_p_bits}b')
        y_binary.append([int(bit) for bit in p_binary])
    
    return np.array(X_binary, dtype=np.float32), np.array(y_binary, dtype=np.float32)


class BinarySemiprimeDataset(Dataset):
    """Pure binary dataset following papers exactly."""
    
    def __init__(self, X_binary, y_binary):
        self.X = torch.FloatTensor(X_binary)
        self.y = torch.FloatTensor(y_binary)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BinaryLSTM(nn.Module):
    """Binary LSTM exactly matching Murat et al. architecture."""
    
    def __init__(self, input_bits=20, output_bits=10, dropout_rate=0.3):
        super(BinaryLSTM, self).__init__()
        
        # Murat et al. architecture: 3 LSTM layers (128, 256, 512)
        self.lstm1 = nn.LSTM(1, 128, batch_first=True)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lstm2 = nn.LSTM(128, 256, batch_first=True)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.lstm3 = nn.LSTM(256, 512, batch_first=True)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Dense layers (128, 100)
        self.dense1 = nn.Linear(512, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.dense2 = nn.Linear(128, 100)
        self.bn5 = nn.BatchNorm1d(100)
        self.dropout5 = nn.Dropout(0.4)
        
        # Output layer - sigmoid for binary classification
        self.output = nn.Linear(100, output_bits)
        
    def forward(self, x):
        # Reshape for LSTM: (batch, sequence_length, input_size)
        x = x.unsqueeze(2)  # (batch, bits, 1)
        
        # LSTM layers with batch norm and dropout
        x, _ = self.lstm1(x)
        x = self.bn1(x[:, -1, :])  # Take last output
        x = self.dropout1(x)
        
        x = x.unsqueeze(1)
        x, _ = self.lstm2(x)
        x = self.bn2(x[:, -1, :])
        x = self.dropout2(x)
        
        x = x.unsqueeze(1)
        x, _ = self.lstm3(x)
        x = self.bn3(x[:, -1, :])
        x = self.dropout3(x)
        
        # Dense layers
        x = torch.relu(self.dense1(x))
        x = self.bn4(x)
        x = self.dropout4(x)
        
        x = torch.relu(self.dense2(x))
        x = self.bn5(x)
        x = self.dropout5(x)
        
        # Output with sigmoid for binary classification
        x = torch.sigmoid(self.output(x))
        return x


def calculate_beta_metrics(predictions, targets):
    """Calculate β_i metrics from papers."""
    # Convert to binary predictions
    binary_preds = (predictions > 0.5).float()
    
    # Count bit errors for each sample
    errors = (binary_preds != targets).sum(dim=1)
    
    beta_metrics = {}
    for i in range(5):
        beta_metrics[f'beta_{i}'] = (errors <= i).float().mean().item()
    
    return beta_metrics


def train_kaggle_binary_model(dataset_name="medium", epochs=120, batch_size=4):
    """Main training function for Kaggle environment."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets - adjust paths for Kaggle
    if os.path.exists(f"/kaggle/input"):
        # Kaggle environment
        base_path = "/kaggle/input/rsa-factorization-data"  # Update this to your dataset name
        train_path = f"{base_path}/{dataset_name}_train.csv"
        test_path = f"{base_path}/{dataset_name}_test.csv"
    else:
        # Local environment
        train_path = f"data/{dataset_name}_train.csv"
        test_path = f"data/{dataset_name}_test.csv"
    
    if not os.path.exists(train_path):
        print(f"Dataset not found at {train_path}")
        print("Please upload your data folder to Kaggle as a dataset")
        return
    
    # Load CSV data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded {len(train_df)} training samples, {len(test_df)} test samples")
    print(f"Max N: {max(train_df['N'].max(), test_df['N'].max()):,}")
    print(f"Max p: {max(train_df['p'].max(), test_df['p'].max()):,}")
    
    # Calculate bit sizes
    max_N = max(train_df['N'].max(), test_df['N'].max())
    max_p = max(train_df['p'].max(), test_df['p'].max())
    max_N_bits = int(max_N).bit_length()
    max_p_bits = int(max_p).bit_length()
    
    print(f"Using {max_N_bits} bits for N, {max_p_bits} bits for p")
    
    # Convert to binary vectors
    X_train_binary, y_train_binary = convert_to_binary_vectors(
        train_df['N'].values, train_df['p'].values, max_N_bits, max_p_bits)
    X_test_binary, y_test_binary = convert_to_binary_vectors(
        test_df['N'].values, test_df['p'].values, max_N_bits, max_p_bits)
    
    # Create datasets
    train_dataset = BinarySemiprimeDataset(X_train_binary, y_train_binary)
    test_dataset = BinarySemiprimeDataset(X_test_binary, y_test_binary)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = BinaryLSTM(max_N_bits, max_p_bits).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    
    print(f"Training Binary LSTM with {max_N_bits}-bit input, {max_p_bits}-bit output")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    model.train()
    training_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate every 10 epochs for Kaggle
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            test_loss = 0
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    test_loss += criterion(outputs, batch_y).item()
                    
                    all_predictions.append(outputs)
                    all_targets.append(batch_y)
            
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Calculate β metrics
            beta_metrics = calculate_beta_metrics(all_predictions, all_targets)
            avg_test_loss = test_loss / len(test_loader)
            
            print(f"Epoch {epoch+1:3d}: Train Loss={avg_loss:.4f}, Test Loss={avg_test_loss:.4f}")
            print(f"         β₀={beta_metrics['beta_0']:.4f}, β₁={beta_metrics['beta_1']:.4f}, "
                  f"β₂={beta_metrics['beta_2']:.4f}")
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'test_loss': avg_test_loss,
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
            outputs = model(batch_x)
            final_predictions.append(outputs)
            final_targets.append(batch_y)
    
    final_predictions = torch.cat(final_predictions, dim=0)
    final_targets = torch.cat(final_targets, dim=0)
    final_beta_metrics = calculate_beta_metrics(final_predictions, final_targets)
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS - {dataset_name.upper()} DATASET")
    print(f"{'='*50}")
    print(f"β₀ (Exact Match): {final_beta_metrics['beta_0']:.4f}")
    print(f"β₁ (≤1 bit error): {final_beta_metrics['beta_1']:.4f}")
    print(f"β₂ (≤2 bit error): {final_beta_metrics['beta_2']:.4f}")
    print(f"β₃ (≤3 bit error): {final_beta_metrics['beta_3']:.4f}")
    print(f"β₄ (≤4 bit error): {final_beta_metrics['beta_4']:.4f}")
    
    # Save results
    results_dir = f"kaggle_results_{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save training history
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(f"{results_dir}/training_history.csv", index=False)
    print(f"Training history saved to: {results_dir}/training_history.csv")
    
    # Save final metrics
    with open(f"{results_dir}/final_metrics.json", 'w') as f:
        json.dump(final_beta_metrics, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), f"{results_dir}/binary_lstm_model.pth")
    print(f"Model saved to: {results_dir}/binary_lstm_model.pth")
    
    return model, training_history, final_beta_metrics


# For Kaggle/Colab - run directly without argparse
if __name__ == "__main__":
    # You can modify these parameters directly
    DATASET = "medium"  # Change to "small", "large", etc.
    EPOCHS = 120
    BATCH_SIZE = 32    # Larger batch size for GPU
    
    print(f"Starting Kaggle Binary Training: {DATASET} dataset, {EPOCHS} epochs")
    
    model, history, metrics = train_kaggle_binary_model(
        dataset_name=DATASET,
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE
    )
    
    print("Training completed successfully!")