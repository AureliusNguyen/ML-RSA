"""
Baseline LSTM model implementing Murat et al.'s architecture.
This serves as our reproduction baseline before adding novel components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional


class SemiprimeDataset(Dataset):
    """Dataset for semiprime factorization task."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BaselineLSTM(nn.Module):
    """
    Baseline LSTM model reproducing Murat et al.'s architecture.
    
    Architecture:
    - 3 LSTM layers: 128, 256, 512 units
    - Batch normalization and dropout after each layer
    - 2 dense layers: 128, 100 units
    - Binary cross entropy loss for bit prediction
    """
    
    def __init__(self, input_size: int = 47, output_size: int = 15, dropout_rate: float = 0.3):
        super(BaselineLSTM, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True, dropout=dropout_rate)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lstm2 = nn.LSTM(128, 256, batch_first=True, dropout=dropout_rate)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.lstm3 = nn.LSTM(256, 512, batch_first=True)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Dense layers
        self.fc1 = nn.Linear(512, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 100)
        self.bn5 = nn.BatchNorm1d(100)
        self.dropout5 = nn.Dropout(0.4)  # Higher dropout for final layer
        
        self.output_layer = nn.Linear(100, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x = x.squeeze(1) if x.size(1) == 1 else x[:, -1, :]  # Take last output
        x = self.bn1(x)
        x = self.dropout1(x)
        x = x.unsqueeze(1)
        
        x, _ = self.lstm2(x)
        x = x.squeeze(1) if x.size(1) == 1 else x[:, -1, :]
        x = self.bn2(x)
        x = self.dropout2(x)
        x = x.unsqueeze(1)
        
        x, _ = self.lstm3(x)
        x = x.squeeze(1) if x.size(1) == 1 else x[:, -1, :]
        x = self.bn3(x)
        x = self.dropout3(x)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.bn4(x)
        x = self.dropout4(x)
        
        x = F.relu(self.fc2(x))
        x = self.bn5(x)
        x = self.dropout5(x)
        
        # Output with sigmoid activation for binary prediction
        x = torch.sigmoid(self.output_layer(x))
        
        return x


class LSTMTrainer:
    """Training utilities for LSTM model."""
    
    def __init__(self, model: BaselineLSTM, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate model performance."""
        self.model.eval()
        total_loss = 0.0
        correct_bits = 0
        total_bits = 0
        exact_matches = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                # Calculate bit accuracy
                predictions = (outputs > 0.5).float()
                correct_bits += (predictions == batch_y).sum().item()
                total_bits += batch_y.numel()
                
                # Calculate exact match accuracy (β0 metric from papers)
                exact_matches += (predictions == batch_y).all(dim=1).sum().item()
                total_samples += batch_y.size(0)
        
        avg_loss = total_loss / len(dataloader)
        bit_accuracy = correct_bits / total_bits
        exact_accuracy = exact_matches / total_samples
        
        return avg_loss, bit_accuracy, exact_accuracy
    
    def calculate_beta_metrics(self, dataloader: DataLoader) -> dict:
        """
        Calculate β_i metrics as used in Murat et al.
        β_i = percentage of predictions with at most i bit errors
        """
        self.model.eval()
        beta_metrics = {f'beta_{i}': 0 for i in range(5)}
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                predictions = (outputs > 0.5).float()
                
                # Count errors for each sample
                errors = (predictions != batch_y).sum(dim=1)
                
                for i in range(5):
                    beta_metrics[f'beta_{i}'] += (errors <= i).sum().item()
                
                total_samples += batch_y.size(0)
        
        # Convert to percentages
        for key in beta_metrics:
            beta_metrics[key] = beta_metrics[key] / total_samples
        
        return beta_metrics


def train_baseline_model(X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray, y_test: np.ndarray,
                        epochs: int = 120, batch_size: int = 4) -> BaselineLSTM:
    """
    Train the baseline LSTM model with specified parameters.
    """
    print("Initializing baseline LSTM training...")
    
    # Create datasets
    train_dataset = SemiprimeDataset(X_train, y_train)
    test_dataset = SemiprimeDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and trainer
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    model = BaselineLSTM(input_size, output_size)
    trainer = LSTMTrainer(model)
    
    print(f"Model architecture: {input_size} -> LSTM(128,256,512) -> Dense(128,100) -> {output_size}")
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    
    # Training loop
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        
        if (epoch + 1) % 20 == 0:  # Evaluate every 20 epochs
            test_loss, bit_acc, exact_acc = trainer.evaluate(test_loader)
            beta_metrics = trainer.calculate_beta_metrics(test_loader)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Bit Accuracy: {bit_acc:.4f}")
            print(f"  Exact Match (β0): {exact_acc:.4f}")
            print(f"  β1: {beta_metrics['beta_1']:.4f}")
            print(f"  β2: {beta_metrics['beta_2']:.4f}")
    
    return model, trainer


if __name__ == "__main__":
    from ..crypto_utils import generate_training_dataset
    
    # Generate test data
    print("Generating training dataset...")
    X, y = generate_training_dataset(5000, 20)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    model, trainer = train_baseline_model(X_train, y_train, X_test, y_test, epochs=50)
    print("Baseline training completed!")