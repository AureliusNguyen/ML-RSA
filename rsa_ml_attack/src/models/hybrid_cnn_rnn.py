"""
Hybrid CNN+RNN model for RSA semiprime factorization.
Combines convolutional pattern recognition with recurrent sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class MathematicalConv1D(nn.Module):
    """
    1D Convolutional layer designed for mathematical feature extraction.
    Uses multiple kernel sizes to capture different mathematical patterns.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list = [3, 5, 7, 9]):
        super(MathematicalConv1D, self).__init__()
        
        self.convolutions = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // len(kernel_sizes), 
                     kernel_size=k, padding=k//2) 
            for k in kernel_sizes
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(out_channels // len(kernel_sizes)) 
            for _ in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Apply each convolution and batch norm
        conv_outputs = []
        for conv, bn in zip(self.convolutions, self.batch_norms):
            conv_out = F.relu(bn(conv(x)))
            conv_outputs.append(conv_out)
        
        # Concatenate outputs from different kernel sizes
        combined = torch.cat(conv_outputs, dim=1)
        return self.dropout(combined)


class FeatureSpecificCNN(nn.Module):
    """
    CNN component that processes different feature types separately.
    Designed to capture local patterns in binary, ECPP, and GNFS features.
    """
    
    def __init__(self, feature_sizes: dict, cnn_channels: list = [64, 128, 256]):
        super(FeatureSpecificCNN, self).__init__()
        
        self.feature_sizes = feature_sizes
        
        # Separate CNN branches for different feature types
        self.binary_cnn = self._create_cnn_branch(1, cnn_channels, feature_sizes['binary'])
        self.ecpp_cnn = self._create_cnn_branch(1, cnn_channels, feature_sizes['ecpp'])
        self.gnfs_cnn = self._create_cnn_branch(1, cnn_channels, feature_sizes['gnfs'])
        
        # Feature fusion layer
        total_features = 3 * cnn_channels[-1]  # 3 feature types
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, cnn_channels[-1]),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels[-1]),
            nn.Dropout(0.3)
        )
    
    def _create_cnn_branch(self, in_channels: int, out_channels: list, seq_length: int):
        """Create CNN branch for specific feature type."""
        layers = []
        
        current_channels = in_channels
        current_length = seq_length
        
        for out_ch in out_channels:
            # Multi-scale convolution
            layers.append(MathematicalConv1D(current_channels, out_ch))
            
            # Adaptive pooling to reduce sequence length
            if current_length > 4:  # Don't pool if sequence is too short
                layers.append(nn.AdaptiveMaxPool1d(max(current_length // 2, 4)))
                current_length = max(current_length // 2, 4)
            
            current_channels = out_ch
        
        # Global pooling to get fixed-size output
        layers.append(nn.AdaptiveAvgPool1d(1))
        
        return nn.Sequential(*layers)
    
    def forward(self, binary_features, ecpp_features, gnfs_features):
        # Add channel dimension and process each feature type
        binary_input = binary_features.unsqueeze(1)  # [batch, 1, seq]
        ecpp_input = ecpp_features.unsqueeze(1)
        gnfs_input = gnfs_features.unsqueeze(1)
        
        # Process through separate CNN branches
        binary_out = self.binary_cnn(binary_input).squeeze(-1)  # Remove sequence dim
        ecpp_out = self.ecpp_cnn(ecpp_input).squeeze(-1)
        gnfs_out = self.gnfs_cnn(gnfs_input).squeeze(-1)
        
        # Concatenate and fuse features
        combined = torch.cat([binary_out, ecpp_out, gnfs_out], dim=1)
        fused = self.fusion_layer(combined)
        
        return fused


class EnhancedLSTM(nn.Module):
    """
    Enhanced LSTM with mathematical attention mechanism.
    Builds on the baseline LSTM with attention and residual connections.
    """
    
    def __init__(self, input_size: int, hidden_sizes: list = [256, 512, 256], 
                 dropout: float = 0.3, num_attention_heads: int = 8):
        super(EnhancedLSTM, self).__init__()
        
        # LSTM layers with different hidden sizes
        self.lstm_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        current_size = input_size
        for hidden_size in hidden_sizes:
            self.lstm_layers.append(
                nn.LSTM(current_size, hidden_size, batch_first=True, dropout=dropout if len(hidden_sizes) > 1 else 0)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            self.dropouts.append(nn.Dropout(dropout))
            current_size = hidden_size
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_sizes[-1],
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention_norm = nn.LayerNorm(hidden_sizes[-1])
        self.output_size = hidden_sizes[-1]
    
    def forward(self, x):
        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, features]
        
        # Pass through LSTM layers
        for lstm, bn, dropout in zip(self.lstm_layers, self.batch_norms, self.dropouts):
            lstm_out, _ = lstm(x)
            
            # Apply batch norm to the last timestep output
            if lstm_out.size(1) == 1:
                normed = bn(lstm_out.squeeze(1)).unsqueeze(1)
            else:
                # For multi-timestep sequences
                batch_size, seq_len, hidden_size = lstm_out.shape
                normed = bn(lstm_out.reshape(-1, hidden_size)).reshape(batch_size, seq_len, hidden_size)
            
            x = dropout(normed)
        
        # Apply self-attention
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.attention_norm(x + attn_out)  # Residual connection
        
        # Return last timestep output
        return x.squeeze(1) if x.size(1) == 1 else x[:, -1, :], attn_weights


class HybridCNNRNN(nn.Module):
    """
    Hybrid CNN+RNN model for RSA semiprime factorization.
    
    Architecture:
    1. Feature-specific CNNs extract local patterns
    2. Enhanced LSTM captures sequential dependencies  
    3. Mathematical fusion combines both representations
    4. Multi-head prediction for robust factor generation
    """
    
    def __init__(self, 
                 feature_sizes: dict,
                 cnn_channels: list = [64, 128, 256],
                 lstm_hidden: list = [256, 512, 256],
                 output_size: int = 15,
                 num_prediction_heads: int = 3):
        super(HybridCNNRNN, self).__init__()
        
        self.feature_sizes = feature_sizes
        self.output_size = output_size
        self.num_prediction_heads = num_prediction_heads
        
        # CNN component for local pattern recognition
        self.cnn_component = FeatureSpecificCNN(feature_sizes, cnn_channels)
        cnn_output_size = cnn_channels[-1]
        
        # RNN component for sequence modeling
        total_input_size = sum(feature_sizes.values())
        self.rnn_component = EnhancedLSTM(total_input_size, lstm_hidden)
        rnn_output_size = lstm_hidden[-1]
        
        # Feature fusion and enhancement
        combined_size = cnn_output_size + rnn_output_size
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_size, combined_size),
            nn.ReLU(),
            nn.BatchNorm1d(combined_size),
            nn.Dropout(0.3),
            nn.Linear(combined_size, combined_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(combined_size // 2),
            nn.Dropout(0.3)
        )
        
        # Mathematical constraint layer
        self.constraint_layer = nn.Sequential(
            nn.Linear(combined_size // 2, combined_size // 4),
            nn.ReLU(),
            nn.LayerNorm(combined_size // 4)
        )
        
        # Multiple prediction heads for ensemble-like behavior
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined_size // 4, output_size * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(output_size * 2, output_size),
                nn.Sigmoid()
            ) for _ in range(num_prediction_heads)
        ])
        
        # Head aggregation
        self.head_weights = nn.Parameter(torch.ones(num_prediction_heads) / num_prediction_heads)
        
    def forward(self, binary_features, ecpp_features, gnfs_features):
        # CNN pathway: Extract local patterns
        cnn_features = self.cnn_component(binary_features, ecpp_features, gnfs_features)
        
        # RNN pathway: Model sequential dependencies
        combined_input = torch.cat([binary_features, ecpp_features, gnfs_features], dim=1)
        rnn_features, attention_weights = self.rnn_component(combined_input)
        
        # Fuse CNN and RNN representations
        combined_features = torch.cat([cnn_features, rnn_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # Apply mathematical constraints
        constrained_features = self.constraint_layer(fused_features)
        
        # Generate predictions from multiple heads
        head_predictions = []
        for head in self.prediction_heads:
            pred = head(constrained_features)
            head_predictions.append(pred)
        
        # Weighted aggregation of predictions
        stacked_predictions = torch.stack(head_predictions, dim=1)  # [batch, num_heads, output_size]
        weights = F.softmax(self.head_weights, dim=0)
        
        final_prediction = torch.sum(stacked_predictions * weights.unsqueeze(0).unsqueeze(-1), dim=1)
        
        # Ensure last bit is 1 (odd number constraint)
        final_prediction[:, -1] = torch.sigmoid(final_prediction[:, -1] + 1.0)
        
        return final_prediction, attention_weights, head_predictions


class HybridTrainer:
    """Training utilities for Hybrid CNN+RNN model."""
    
    def __init__(self, model: HybridCNNRNN, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Enhanced optimizer with different learning rates for different components
        param_groups = [
            {'params': self.model.cnn_component.parameters(), 'lr': 0.001},
            {'params': self.model.rnn_component.parameters(), 'lr': 0.0005},
            {'params': self.model.fusion_layer.parameters(), 'lr': 0.001},
            {'params': self.model.prediction_heads.parameters(), 'lr': 0.001},
        ]
        
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        
        # Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # Enhanced loss function
        self.criterion = self._create_enhanced_loss()
        
    def _create_enhanced_loss(self):
        """Create sophisticated loss function with mathematical constraints."""
        class EnhancedFactorizationLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.bce = nn.BCELoss()
                self.mse = nn.MSELoss()
            
            def forward(self, predictions, targets, head_predictions=None):
                # Primary BCE loss
                bce_loss = self.bce(predictions, targets)
                
                # Mathematical constraint losses
                constraints_loss = 0.0
                
                # Odd number constraint (last bit should be 1)
                odd_constraint = torch.mean((1.0 - predictions[:, -1]) ** 2)
                constraints_loss += 0.1 * odd_constraint
                
                # Consistency across prediction heads
                if head_predictions is not None:
                    head_consistency = 0.0
                    for i in range(len(head_predictions)):
                        for j in range(i+1, len(head_predictions)):
                            head_consistency += self.mse(head_predictions[i], head_predictions[j])
                    constraints_loss += 0.05 * head_consistency / (len(head_predictions) * (len(head_predictions) - 1) / 2)
                
                # Bit transition smoothness (encourage coherent bit patterns)
                bit_transitions = torch.abs(predictions[:, 1:] - predictions[:, :-1])
                smoothness_loss = torch.mean(bit_transitions)
                constraints_loss += 0.02 * smoothness_loss
                
                total_loss = bce_loss + constraints_loss
                return total_loss
        
        return EnhancedFactorizationLoss()
    
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch with enhanced logging."""
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch_binary, batch_ecpp, batch_gnfs, batch_y in dataloader:
            batch_binary = batch_binary.to(self.device)
            batch_ecpp = batch_ecpp.to(self.device)
            batch_gnfs = batch_gnfs.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions, attention_weights, head_predictions = self.model(
                batch_binary, batch_ecpp, batch_gnfs
            )
            
            loss = self.criterion(predictions, batch_y, head_predictions)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        # Update learning rate
        self.scheduler.step()
        
        return total_loss / batch_count
    
    def evaluate_with_ensemble(self, dataloader) -> dict:
        """Evaluate model with ensemble predictions from multiple heads."""
        self.model.eval()
        total_loss = 0.0
        
        # Metrics tracking
        exact_matches = 0
        head_exact_matches = [0] * self.model.num_prediction_heads
        total_samples = 0
        
        with torch.no_grad():
            for batch_binary, batch_ecpp, batch_gnfs, batch_y in dataloader:
                batch_binary = batch_binary.to(self.device)
                batch_ecpp = batch_ecpp.to(self.device)
                batch_gnfs = batch_gnfs.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions, _, head_predictions = self.model(
                    batch_binary, batch_ecpp, batch_gnfs
                )
                
                loss = self.criterion(predictions, batch_y, head_predictions)
                total_loss += loss.item()
                
                # Calculate ensemble accuracy
                binary_preds = (predictions > 0.5).float()
                exact_matches += (binary_preds == batch_y).all(dim=1).sum().item()
                
                # Calculate individual head accuracies
                for i, head_pred in enumerate(head_predictions):
                    head_binary = (head_pred > 0.5).float()
                    head_exact_matches[i] += (head_binary == batch_y).all(dim=1).sum().item()
                
                total_samples += batch_y.size(0)
        
        results = {
            'loss': total_loss / len(dataloader),
            'ensemble_accuracy': exact_matches / total_samples,
            'head_accuracies': [matches / total_samples for matches in head_exact_matches],
            'best_head_accuracy': max([matches / total_samples for matches in head_exact_matches])
        }
        
        return results


if __name__ == "__main__":
    # Test hybrid model
    feature_sizes = {'binary': 30, 'ecpp': 44, 'gnfs': 46}
    model = HybridCNNRNN(feature_sizes)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Hybrid CNN+RNN model created with {total_params:,} parameters")
    
    # Test forward pass
    batch_size = 16
    binary_test = torch.randn(batch_size, 30)
    eccp_test = torch.randn(batch_size, 44)
    gnfs_test = torch.randn(batch_size, 46)
    
    with torch.no_grad():
        output, attention, heads = model(binary_test, eccp_test, gnfs_test)
        print(f"Output shape: {output.shape}")
        print(f"Number of prediction heads: {len(heads)}")
        print(f"Attention weights shape: {attention.shape if attention is not None else 'None'}")
        print(f"Sample output: {output[0][:10]}")  # First 10 bits of first sample