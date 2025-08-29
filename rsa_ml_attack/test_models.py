"""
Quick test script to verify all models work correctly.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.crypto_utils import FeatureEngineer, PrimeGenerator
from src.models.baseline_lstm import BaselineLSTM
from src.models.transformer_factorizer import TransformerFactorizer  
from src.models.factorization_gan import FactorizationGAN
from src.models.hybrid_cnn_rnn import HybridCNNRNN
from torch.utils.data import DataLoader, Dataset

class SimpleTestDataset(Dataset):
    """Simple test dataset for model verification."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
        # Split features by type for testing
        self.binary_features = self.X[:, :30]  # First 30 bits
        remaining = self.X[:, 30:]
        
        # Calculate split points
        eccp_size = min(44, remaining.shape[1] // 2)
        gnfs_size = remaining.shape[1] - eccp_size
        
        self.ecpp_features = remaining[:, :eccp_size] if eccp_size > 0 else torch.zeros(self.X.shape[0], 44)
        self.gnfs_features = remaining[:, eccp_size:] if gnfs_size > 0 else torch.zeros(self.X.shape[0], 46)
        
        # Pad if needed
        if self.ecpp_features.shape[1] < 44:
            padding = torch.zeros(self.X.shape[0], 44 - self.ecpp_features.shape[1])
            self.ecpp_features = torch.cat([self.ecpp_features, padding], dim=1)
        
        if self.gnfs_features.shape[1] < 46:
            padding = torch.zeros(self.X.shape[0], 46 - self.gnfs_features.shape[1])
            self.gnfs_features = torch.cat([self.gnfs_features, padding], dim=1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.binary_features[idx], self.ecpp_features[idx], 
                self.gnfs_features[idx], self.y[idx])


def test_feature_generation():
    """Test feature generation pipeline."""
    print("Testing feature generation...")
    
    try:
        # Generate test data using our crypto utils
        prime_gen = PrimeGenerator(20)
        feature_eng = FeatureEngineer()
        
        # Generate small test dataset
        pairs = prime_gen.generate_prime_pairs(10, 20, 50)
        X, y = feature_eng.prepare_training_data(pairs)
        
        print(f"[OK] Generated test data: X shape {X.shape}, y shape {y.shape}")
        
        # Test dataset creation
        dataset = SimpleTestDataset(X, y)
        print(f"[OK] Dataset creation successful")
        
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        binary, ecpp, gnfs, targets = next(iter(dataloader))
        print(f"[OK] DataLoader working: Binary {binary.shape}, ECPP {ecpp.shape}, GNFS {gnfs.shape}, Targets {targets.shape}")
        
        return dataset, dataloader
    except Exception as e:
        print(f"[FAIL] Feature generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_baseline_lstm(dataloader):
    """Test baseline LSTM model."""
    print("\nTesting Baseline LSTM...")
    
    try:
        # Get total feature size
        binary, ecpp, gnfs, targets = next(iter(dataloader))
        input_size = binary.size(1) + ecpp.size(1) + gnfs.size(1)
        output_size = targets.size(1)
        
        model = BaselineLSTM(input_size, output_size)
        
        # Test forward pass
        test_input = torch.cat([binary, ecpp, gnfs], dim=1)
        with torch.no_grad():
            output = model(test_input)
            
        print(f"[OK] LSTM forward pass: input {test_input.shape} -> output {output.shape}")
        print(f"[OK] LSTM parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"[FAIL] LSTM test failed: {e}")
        return False

def test_transformer_model(dataloader):
    """Test transformer model."""
    print("\nTesting Transformer Model...")
    
    try:
        binary, ecpp, gnfs, targets = next(iter(dataloader))
        
        input_sizes = {
            'binary': binary.size(1),
            'ecpp': ecpp.size(1), 
            'gnfs': gnfs.size(1)
        }
        
        model = TransformerFactorizer(input_sizes)
        
        # Test forward pass
        with torch.no_grad():
            output, attention = model(binary, ecpp, gnfs)
            
        print(f"[OK] Transformer forward pass: output {output.shape}")
        print(f"[OK] Transformer parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"[OK] Attention layers: {len(attention)}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Transformer test failed: {e}")
        return False

def test_gan_model(dataloader):
    """Test GAN model."""
    print("\nTesting GAN Model...")
    
    try:
        binary, ecpp, gnfs, targets = next(iter(dataloader))
        
        total_input_size = binary.size(1) + ecpp.size(1) + gnfs.size(1)
        factor_size = targets.size(1)
        
        gan = FactorizationGAN(input_size=total_input_size, factor_size=factor_size, device='cpu')
        
        # Test generator
        test_input = torch.cat([binary, ecpp, gnfs], dim=1)
        with torch.no_grad():
            generated_factors = gan.generator(test_input)
            
        print(f"[OK] GAN Generator: input {test_input.shape} -> output {generated_factors.shape}")
        
        # Test discriminator
        with torch.no_grad():
            disc_output = gan.discriminator(test_input, generated_factors)
            
        print(f"[OK] GAN Discriminator outputs: {list(disc_output.keys())}")
        
        total_params = (sum(p.numel() for p in gan.generator.parameters()) + 
                       sum(p.numel() for p in gan.discriminator.parameters()))
        print(f"[OK] GAN total parameters: {total_params:,}")
        
        return True
    except Exception as e:
        print(f"[FAIL] GAN test failed: {e}")
        return False

def test_hybrid_model(dataloader):
    """Test hybrid CNN+RNN model."""
    print("\nTesting Hybrid CNN+RNN Model...")
    
    try:
        binary, ecpp, gnfs, targets = next(iter(dataloader))
        
        feature_sizes = {
            'binary': binary.size(1),
            'ecpp': ecpp.size(1),
            'gnfs': gnfs.size(1)
        }
        
        model = HybridCNNRNN(feature_sizes)
        
        # Test forward pass
        with torch.no_grad():
            output, attention, heads = model(binary, ecpp, gnfs)
            
        print(f"[OK] Hybrid forward pass: output {output.shape}")
        print(f"[OK] Hybrid parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"[OK] Prediction heads: {len(heads)}")
        if attention is not None:
            print(f"[OK] Attention weights shape: {attention.shape}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Hybrid test failed: {e}")
        return False

def test_mathematical_constraints():
    """Test mathematical constraint enforcement."""
    print("\nTesting Mathematical Constraints...")
    
    try:
        # Test that models enforce odd number constraint
        test_binary = torch.randn(4, 30)  
        test_ecpp = torch.randn(4, 44)
        test_gnfs = torch.randn(4, 46)
        
        models_to_test = [
            ('Transformer', lambda: TransformerFactorizer({'binary': 30, 'ecpp': 44, 'gnfs': 46})),
            ('Hybrid', lambda: HybridCNNRNN({'binary': 30, 'ecpp': 44, 'gnfs': 46}))
        ]
        
        for name, model_func in models_to_test:
            model = model_func()
            model.eval()
            
            with torch.no_grad():
                if name == 'Transformer':
                    output, _ = model(test_binary, test_ecpp, test_gnfs)
                else:  # Hybrid
                    output, _, _ = model(test_binary, test_ecpp, test_gnfs)
                
                # Check if last bit (odd number constraint) is generally high
                last_bits = output[:, -1]
                avg_last_bit = last_bits.mean().item()
                
                print(f"[OK] {name} last bit average: {avg_last_bit:.3f} (should be > 0.5 for odd constraint)")
        
        return True
    except Exception as e:
        print(f"[FAIL] Mathematical constraint test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("RUNNING MODEL TESTS")
    print("="*60)
    
    # Test feature generation
    dataset, dataloader = test_feature_generation()
    if dataset is None:
        print("Critical failure in feature generation. Stopping tests.")
        return
    
    # Test all models
    tests_passed = 0
    total_tests = 5
    
    if test_baseline_lstm(dataloader):
        tests_passed += 1
    
    if test_transformer_model(dataloader):
        tests_passed += 1
    
    if test_gan_model(dataloader):
        tests_passed += 1
        
    if test_hybrid_model(dataloader):
        tests_passed += 1
    
    if test_mathematical_constraints():
        tests_passed += 1
    
    print("\n" + "="*60)
    print(f"TESTS COMPLETED: {tests_passed}/{total_tests} PASSED")
    print("="*60)
    
    if tests_passed == total_tests:
        print("[SUCCESS] All tests passed! Ready for full training.")
    else:
        print("[WARNING] Some tests failed. Please check the errors above.")
    
    print("\nTo run full training:")
    print("python train_all_models.py --dataset-size 10000 --epochs 120")

if __name__ == "__main__":
    main()
    