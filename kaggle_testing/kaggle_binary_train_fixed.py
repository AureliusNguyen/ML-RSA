"""
Kaggle Binary Framework Training - Fixed Path Detection
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import json

# Check available paths in Kaggle
print("Available paths:")
if os.path.exists("/kaggle/input"):
    print("Kaggle input directory contents:")
    for item in os.listdir("/kaggle/input"):
        print(f"  /kaggle/input/{item}")
        if os.path.isdir(f"/kaggle/input/{item}"):
            try:
                sub_items = os.listdir(f"/kaggle/input/{item}")
                for sub_item in sub_items:
                    print(f"    {sub_item}")
            except:
                pass

# Binary conversion utilities
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
        
        # Output layer
        self.output = nn.Linear(100, output_bits)
        
    def forward(self, x):
        # Reshape for LSTM
        x = x.unsqueeze(2)  # (batch, bits, 1)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.bn1(x[:, -1, :])
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
        
        # Output with sigmoid
        x = torch.sigmoid(self.output(x))
        return x


def calculate_beta_metrics(predictions, targets):
    """Calculate β_i metrics from papers."""
    binary_preds = (predictions > 0.5).float()
    errors = (binary_preds != targets).sum(dim=1)
    
    beta_metrics = {}
    for i in range(5):
        beta_metrics[f'beta_{i}'] = (errors <= i).float().mean().item()
    
    return beta_metrics


def find_dataset_files(dataset_name):
    """Find dataset files in Kaggle environment."""
    # Try multiple possible paths
    possible_paths = [
        f"/kaggle/input/rsa-factorization-data/{dataset_name}_train.csv",
        f"/kaggle/input/rsa-ml-attack-data/{dataset_name}_train.csv",
        f"/kaggle/input/{dataset_name}_train.csv"
    ]
    
    # Also check all subdirectories
    if os.path.exists("/kaggle/input"):
        for item in os.listdir("/kaggle/input"):
            item_path = f"/kaggle/input/{item}"
            if os.path.isdir(item_path):
                possible_paths.append(f"{item_path}/{dataset_name}_train.csv")
                possible_paths.append(f"{item_path}/{dataset_name}_test.csv")
    
    # Find the correct paths
    train_path = None
    test_path = None
    
    for path in possible_paths:
        if path.endswith("_train.csv") and os.path.exists(path):
            train_path = path
            test_path = path.replace("_train.csv", "_test.csv")
            break
    
    return train_path, test_path


def train_kaggle_binary_model(dataset_name="medium", epochs=120, batch_size=32):
    """Main training function for Kaggle environment."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find dataset files
    train_path, test_path = find_dataset_files(dataset_name)
    
    if train_path is None or not os.path.exists(train_path):
        print(f"❌ Could not find {dataset_name}_train.csv")
        print(f"❌ Searched paths include:")
        print(f"   /kaggle/input/*/")
        print(f"❌ Please make sure you uploaded the data folder as a Kaggle dataset")
        print(f"❌ and added it to this notebook.")
        return None, None, None  # Return None values to avoid unpack error
    
    print(f"✅ Found dataset files:")
    print(f"   Train: {train_path}")
    print(f"   Test: {test_path}")
    
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
    
    print(f"🚀 Training Binary LSTM: {max_N_bits}-bit input → {max_p_bits}-bit output")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"⚙️  Device: {device}, Batch size: {batch_size}, Epochs: {epochs}")
    
    # Training loop
    model.train()
    training_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{epochs}")
        
        for batch_x, batch_y in progress_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
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
                    outputs = model(batch_x)
                    test_loss += criterion(outputs, batch_y).item()
                    
                    all_predictions.append(outputs)
                    all_targets.append(batch_y)
            
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Calculate β metrics
            beta_metrics = calculate_beta_metrics(all_predictions, all_targets)
            avg_test_loss = test_loss / len(test_loader)
            
            print(f"📈 Epoch {epoch+1:3d}: Train Loss={avg_loss:.4f}, Test Loss={avg_test_loss:.4f}")
            print(f"   β₀={beta_metrics['beta_0']:.4f}, β₁={beta_metrics['beta_1']:.4f}, "
                  f"β₂={beta_metrics['beta_2']:.4f}, β₃={beta_metrics['beta_3']:.4f}")
            
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
    
    print(f"\n🎉 FINAL RESULTS - {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    print(f"β₀ (Exact Match):   {final_beta_metrics['beta_0']:.4f} ({final_beta_metrics['beta_0']*100:.2f}%)")
    print(f"β₁ (≤1 bit error):  {final_beta_metrics['beta_1']:.4f} ({final_beta_metrics['beta_1']*100:.2f}%)")
    print(f"β₂ (≤2 bit error):  {final_beta_metrics['beta_2']:.4f} ({final_beta_metrics['beta_2']*100:.2f}%)")
    print(f"β₃ (≤3 bit error):  {final_beta_metrics['beta_3']:.4f} ({final_beta_metrics['beta_3']*100:.2f}%)")
    print(f"β₄ (≤4 bit error):  {final_beta_metrics['beta_4']:.4f} ({final_beta_metrics['beta_4']*100:.2f}%)")
    
    # Save results
    history_df = pd.DataFrame(training_history)
    print(f"\n💾 Training completed! History shape: {history_df.shape}")
    
    return model, training_history, final_beta_metrics


# Main execution
if __name__ == "__main__":
    DATASET = "medium"  # Change as needed
    EPOCHS = 120
    BATCH_SIZE = 32
    
    print(f"🚀 Starting Kaggle Binary Training")
    print(f"📊 Dataset: {DATASET}, Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")
    
    try:
        model, history, metrics = train_kaggle_binary_model(
            dataset_name=DATASET,
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE
        )
        
        if model is not None:
            print("✅ Training completed successfully!")
        else:
            print("❌ Training failed - check dataset upload")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()