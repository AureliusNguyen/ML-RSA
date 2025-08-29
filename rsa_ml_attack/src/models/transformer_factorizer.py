"""
Transformer-based architecture for RSA semiprime factorization.
Leverages attention mechanisms to capture long-range dependencies in mathematical features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for mathematical feature sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MathematicalAttention(nn.Module):
    """
    Custom attention mechanism that incorporates mathematical structure.
    Weights attention based on mathematical relationships between features.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super(MathematicalAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        
        # Mathematical structure bias
        self.structure_bias = nn.Parameter(torch.randn(num_heads, 1, 1))
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Generate queries, keys, values
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add mathematical structure bias
        scores = scores + self.structure_bias
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return context, attention


class TransformerEncoderBlock(nn.Module):
    """Enhanced transformer encoder block for mathematical features."""
    
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        self.self_attention = MathematicalAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Enhanced feed-forward network with GELU activation
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out, attention_weights = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x, attention_weights


class FeatureEmbedding(nn.Module):
    """Embed different types of mathematical features into unified space."""
    
    def __init__(self, input_sizes: dict, d_model: int):
        super(FeatureEmbedding, self).__init__()
        
        # Check if we're using combined features or separate feature types
        if 'combined' in input_sizes:
            # Single combined feature vector mode
            self.mode = 'combined'
            self.combined_embed = nn.Linear(input_sizes['combined'], d_model)
        else:
            # Separate feature types mode
            self.mode = 'separate'
            # Separate embeddings for different feature types
            self.binary_embed = nn.Linear(input_sizes['binary'], d_model // 3)
            self.ecpp_embed = nn.Linear(input_sizes['ecpp'], d_model // 3)  
            self.gnfs_embed = nn.Linear(input_sizes['gnfs'], d_model - 2 * (d_model // 3))
            
            # Feature type embeddings
            self.type_embeddings = nn.Embedding(3, d_model)  # 3 feature types
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, binary_features=None, ecpp_features=None, gnfs_features=None, combined_features=None):
        if self.mode == 'combined':
            # Combined feature mode - expect single feature tensor
            if combined_features is None:
                # Fallback: try to use first argument as combined features
                combined_features = binary_features
            batch_size = combined_features.size(0)
            
            # Single embedding for combined features
            embeddings = self.combined_embed(combined_features)  # (batch, d_model)
            seq_features = embeddings.unsqueeze(1)  # (batch, 1, d_model) for sequence format
            
        else:
            # Separate feature types mode
            batch_size = binary_features.size(0)
            
            # Embed each feature type
            binary_emb = self.binary_embed(binary_features)
            ecpp_emb = self.ecpp_embed(ecpp_features)
            gnfs_emb = self.gnfs_embed(gnfs_features)
            
            # Concatenate embeddings
            combined = torch.cat([binary_emb, ecpp_emb, gnfs_emb], dim=-1)
            
            # Add positional encodings for each feature type
            seq_features = combined.unsqueeze(1)  # Add sequence dimension
        
        return self.norm(seq_features)


class TransformerFactorizer(nn.Module):
    """
    Transformer-based model for RSA semiprime factorization.
    
    Architecture:
    - Feature embedding layer for different mathematical features
    - Multiple transformer encoder blocks with mathematical attention
    - Factor prediction head with binary output
    """
    
    def __init__(self, 
                 input_sizes: dict,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 output_size: int = 15,
                 dropout: float = 0.1):
        super(TransformerFactorizer, self).__init__()
        
        self.d_model = d_model
        
        # Feature embedding
        self.feature_embedding = FeatureEmbedding(input_sizes, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Mathematical insight integration
        self.math_insight_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Factor prediction head
        self.factor_predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, output_size),
            nn.Sigmoid()
        )
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, binary_features=None, ecpp_features=None, gnfs_features=None, combined_features=None):
        # Embed features - support both combined and separate modes
        if hasattr(self.feature_embedding, 'mode') and self.feature_embedding.mode == 'combined':
            # Use combined_features if provided, otherwise fallback to binary_features
            features_to_use = combined_features if combined_features is not None else binary_features
            embedded = self.feature_embedding(combined_features=features_to_use)
        else:
            embedded = self.feature_embedding(binary_features, eccp_features, gnfs_features)
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded.transpose(0, 1)).transpose(0, 1)
        
        # Pass through transformer encoder layers
        x = embedded
        attention_weights = []
        
        for encoder_layer in self.encoder_layers:
            x, attn_weights = encoder_layer(x)
            attention_weights.append(attn_weights)
        
        # Apply mathematical insights
        x = self.math_insight_layer(x)
        
        # Global pooling (take mean across sequence dimension)
        x = x.mean(dim=1)
        
        # Predict factor bits
        factor_prediction = self.factor_predictor(x)
        
        return factor_prediction, attention_weights


class TransformerTrainer:
    """Training utilities for Transformer factorizer."""
    
    def __init__(self, model: TransformerFactorizer, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Enhanced optimizer with learning rate scheduling
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=0.0001, 
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Enhanced loss function
        self.criterion = self._create_enhanced_loss()
    
    def _create_enhanced_loss(self):
        """Create loss function that incorporates mathematical constraints."""
        class MathematicalBCELoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.bce = nn.BCELoss()
            
            def forward(self, predictions, targets, gnfs_features=None):
                # Basic binary cross-entropy
                bce_loss = self.bce(predictions, targets)
                
                # Mathematical constraint penalty
                # Encourage predictions that satisfy basic mathematical properties
                math_penalty = 0.0
                
                # Primality constraint: predicted factors should be odd (except 2)
                last_bit_penalty = torch.mean((predictions[:, -1] - 1.0) ** 2)  # Last bit should be 1 for odd numbers
                math_penalty += 0.1 * last_bit_penalty
                
                total_loss = bce_loss + math_penalty
                return total_loss
        
        return MathematicalBCELoss()
    
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_binary, batch_ecpp, batch_gnfs, batch_y in dataloader:
            batch_binary = batch_binary.to(self.device)
            batch_ecpp = batch_ecpp.to(self.device)
            batch_gnfs = batch_gnfs.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions, attention_weights = self.model(batch_binary, batch_ecpp, batch_gnfs)
            loss = self.criterion(predictions, batch_y, batch_gnfs)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        self.scheduler.step(avg_loss)
        
        return avg_loss
    
    def evaluate(self, dataloader) -> Tuple[float, float, float]:
        """Evaluate model performance."""
        self.model.eval()
        total_loss = 0.0
        correct_bits = 0
        total_bits = 0
        exact_matches = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_binary, batch_ecpp, batch_gnfs, batch_y in dataloader:
                batch_binary = batch_binary.to(self.device)
                batch_ecpp = batch_ecpp.to(self.device) 
                batch_gnfs = batch_gnfs.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions, _ = self.model(batch_binary, batch_ecpp, batch_gnfs)
                loss = self.criterion(predictions, batch_y, batch_gnfs)
                total_loss += loss.item()
                
                # Calculate metrics
                binary_preds = (predictions > 0.5).float()
                correct_bits += (binary_preds == batch_y).sum().item()
                total_bits += batch_y.numel()
                exact_matches += (binary_preds == batch_y).all(dim=1).sum().item()
                total_samples += batch_y.size(0)
        
        avg_loss = total_loss / len(dataloader)
        bit_accuracy = correct_bits / total_bits
        exact_accuracy = exact_matches / total_samples
        
        return avg_loss, bit_accuracy, exact_accuracy


def create_feature_split_dataloader(X, y, batch_size=32, train_split=0.8):
    """Create dataloaders with separated feature types."""
    from torch.utils.data import Dataset, DataLoader, random_split
    
    class SplitFeatureDataset(Dataset):
        def __init__(self, X, y):
            # Assume features are concatenated: [binary(30), ecpp(?), gnfs(?)]
            self.binary_features = torch.FloatTensor(X[:, :30])  # First 30 bits
            
            # Calculate feature sizes (this should match your feature engineering)
            remaining_features = X[:, 30:]
            ecpp_size = 44  # Approximate from your ECPP implementation
            
            self.ecpp_features = torch.FloatTensor(remaining_features[:, :ecpp_size])
            self.gnfs_features = torch.FloatTensor(remaining_features[:, eccp_size:])
            self.targets = torch.FloatTensor(y)
        
        def __len__(self):
            return len(self.targets)
        
        def __getitem__(self, idx):
            return (self.binary_features[idx], self.ecpp_features[idx], 
                   self.gnfs_features[idx], self.targets[idx])
    
    dataset = SplitFeatureDataset(X, y)
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    input_sizes = {'binary': 30, 'ecpp': 44, 'gnfs': 40}  # Approximate sizes
    model = TransformerFactorizer(input_sizes)
    print(f"Transformer model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 16
    binary_test = torch.randn(batch_size, 30)
    ecpp_test = torch.randn(batch_size, 44)
    gnfs_test = torch.randn(batch_size, 40)
    
    with torch.no_grad():
        output, attention = model(binary_test, eccp_test, gnfs_test)
        print(f"Output shape: {output.shape}")
        print(f"Number of attention layers: {len(attention)}")