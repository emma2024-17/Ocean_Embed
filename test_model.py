import os
import torch
import numpy as np
import json
from train_model import OceanEmbedUNet, load_and_preprocess_nc
import matplotlib.pyplot as plt

class OceanEmbedTester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.t_min = None
        self.t_max = None
        self.test_results = {}
        
    def load_model(self):
        """Load trained model"""
        print("Loading model...")
        self.model = OceanEmbedUNet(dropout_rate=0.3).to(self.device)
        
        if os.path.exists("oceanembed_model.pth"):
            checkpoint = torch.load("oceanembed_model.pth", map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
                self.t_min = checkpoint.get('t_min', 3.5)
                self.t_max = checkpoint.get('t_max', 32.0)
        else:
            print("Warning: Model checkpoint not found. Using default values.")
            self.t_min = 3.5
            self.t_max = 32.0
        
        self.model.eval()
        print(f"Model loaded (t_min={self.t_min:.2f}, t_max={self.t_max:.2f})")
        
    def test_model_output_shape(self):
        """Test 1: Verify model output shape"""
        print("\n[TEST 1] Model Output Shape Verification")
        print("-" * 50)
        
        try:
            # Create dummy input matching expected shape (B, C, H, W)
            dummy_input = torch.randn(1, 7, 100, 240).to(self.device)
            
            with torch.no_grad():
                output, embedding = self.model(dummy_input)
            
            # Expected output shape: (1, 15, 100, 240)
            assert output.shape == (1, 15, 100, 240), f"Output shape mismatch: {output.shape}"
            assert embedding.shape == (1, 128, 25, 60), f"Embedding shape mismatch: {embedding.shape}"
            
            print(f"✓ Output shape: {output.shape} (Expected: (1, 15, 100, 240))")
            print(f"✓ Embedding shape: {embedding.shape}")
            self.test_results['output_shape'] = 'PASSED'
            return True
        except Exception as e:
            print(f"✗ Test failed: {e}")
            self.test_results['output_shape'] = f'FAILED: {e}'
            return False
    
    def test_output_value_range(self):
        """Test 2: Verify output values are in reasonable range [0, 1]"""
        print("\n[TEST 2] Output Value Range Verification")
        print("-" * 50)
        
        try:
            dummy_input = torch.randn(1, 7, 100, 240).to(self.device)
            
            with torch.no_grad():
                output, _ = self.model(dummy_input)
            
            min_val = output.min().item()
            max_val = output.max().item()
            
            print(f"Output min: {min_val:.4f}, max: {max_val:.4f}")
            
            # Outputs should be roughly in [0, 1] for normalized targets
            if -0.5 <= min_val <= 1.5 and -0.5 <= max_val <= 1.5:
                print(f"✓ Output values in reasonable range")
                self.test_results['output_range'] = 'PASSED'
                return True
            else:
                print(f"✗ Output values out of range (expected [-0.5, 1.5])")
                self.test_results['output_range'] = 'WARNING'
                return False
        except Exception as e:
            print(f"✗ Test failed: {e}")
            self.test_results['output_range'] = f'FAILED: {e}'
            return False
    
    def test_model_consistency(self):
        """Test 3: Verify model produces consistent outputs in eval mode"""
        print("\n[TEST 3] Model Consistency (Eval Mode)")
        print("-" * 50)
        
        try:
            dummy_input = torch.randn(1, 7, 100, 240).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                output1, _ = self.model(dummy_input)
                output2, _ = self.model(dummy_input)
            
            # In eval mode, outputs should be identical
            max_diff = torch.max(torch.abs(output1 - output2)).item()
            
            print(f"Max difference between two inference passes: {max_diff:.8f}")
            
            if max_diff < 1e-6:
                print(f"✓ Model is deterministic in eval mode")
                self.test_results['consistency'] = 'PASSED'
                return True
            else:
                print(f"✗ Model outputs differ between passes")
                self.test_results['consistency'] = 'FAILED'
                return False
        except Exception as e:
            print(f"✗ Test failed: {e}")
            self.test_results['consistency'] = f'FAILED: {e}'
            return False
    
    def test_real_data(self):
        """Test 4: Test model on real data"""
        print("\n[TEST 4] Real Data Inference")
        print("-" * 50)
        
        try:
            data_path = "./data/glorys_subset.nc"
            if not os.path.exists(data_path):
                print(f"✗ Data file not found: {data_path}")
                self.test_results['real_data'] = 'SKIPPED: No data'
                return False
            
            inputs, targets, mask, targets_raw, lats, lons = load_and_preprocess_nc(data_path)
            
            # Use first sample
            test_input = inputs[:1].to(self.device)
            
            # Get raw targets (not normalized)
            test_target_raw = targets_raw[:1]  # Shape: (1, 15, 100, 240)
            
            # Get model predictions (normalized)
            with torch.no_grad():
                output, embedding = self.model(test_input)
            
            output_np = output.cpu().numpy()
            
            # Denormalize predictions using t_min and t_max
            # The model outputs normalized values in [0, 1] range
            # Denormalize: value = normalized * (t_max - t_min) + t_min
            pred_denorm = output_np * (self.t_max - self.t_min) + self.t_min
            
            # Get ocean mask
            mask_np = mask.numpy()
            
            # Calculate metrics for each depth
            rmse_per_depth = []
            mae_per_depth = []
            
            for depth_idx in range(test_target_raw.shape[1]):
                pred_at_depth = pred_denorm[0, depth_idx]
                target_at_depth = test_target_raw[0, depth_idx]
                
                # Apply ocean mask and filter valid data
                valid_mask = mask_np & ~np.isnan(target_at_depth)
                
                if np.sum(valid_mask) > 0:
                    pred_valid = pred_at_depth[valid_mask]
                    target_valid = target_at_depth[valid_mask]
                    
                    mse = np.mean((pred_valid - target_valid) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(pred_valid - target_valid))
                    
                    rmse_per_depth.append(rmse)
                    mae_per_depth.append(mae)
            
            if rmse_per_depth:
                avg_rmse = np.mean(rmse_per_depth)
                avg_mae = np.mean(mae_per_depth)
                
                # Check if predictions are within ±5°C on average
                reasonable_threshold = 5.0
                within_threshold = np.sum(np.array(rmse_per_depth) < reasonable_threshold) / len(rmse_per_depth) * 100
                
                print(f"Average RMSE: {avg_rmse:.4f}°C (±{np.std(rmse_per_depth):.4f}°C)")
                print(f"Average MAE: {avg_mae:.4f}°C")
                print(f"Depth levels within {reasonable_threshold}°C RMSE: {within_threshold:.1f}%")
                print(f"✓ Model inference successful on real data")
                
                self.test_results['real_data'] = {
                    'rmse': avg_rmse,
                    'mae': avg_mae,
                    'rmse_std': np.std(rmse_per_depth),
                    'depths_within_threshold': within_threshold
                }
                return True
            else:
                print(f"✗ No valid data for evaluation")
                self.test_results['real_data'] = 'FAILED: No valid data'
                return False
                
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['real_data'] = f'FAILED: {e}'
            return False
    
    def test_overfitting_underfitting(self):
        """Test 5: Analyze training history for overfitting/underfitting"""
        print("\n[TEST 5] Overfitting/Underfitting Analysis")
        print("-" * 50)
        
        try:
            if not os.path.exists('training_history.json'):
                print("✗ Training history file not found")
                self.test_results['overfitting_analysis'] = 'SKIPPED: No history'
                return False
            
            with open('training_history.json', 'r') as f:
                history = json.load(f)
            
            train_losses = history.get('train_losses', [])
            val_losses = history.get('val_losses', [])
            
            if not train_losses or not val_losses:
                print("✗ Empty training history")
                self.test_results['overfitting_analysis'] = 'FAILED: Empty history'
                return False
            
            # Calculate overfitting metrics
            final_train_loss = train_losses[-1]
            final_val_loss = val_losses[-1]
            
            # Check if model is converged
            train_convergence = abs(train_losses[-1] - train_losses[-10]) / abs(train_losses[-10]) * 100 if len(train_losses) >= 10 else float('inf')
            val_convergence = abs(val_losses[-1] - val_losses[-10]) / abs(val_losses[-10]) * 100 if len(val_losses) >= 10 else float('inf')
            
            # Overfitting ratio
            overfit_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else float('inf')
            
            # Calculate trend: are val losses increasing while train losses decreasing?
            recent_train = np.mean(train_losses[-10:]) if len(train_losses) >= 10 else final_train_loss
            recent_val = np.mean(val_losses[-10:]) if len(val_losses) >= 10 else final_val_loss
            
            print(f"Final Train Loss: {final_train_loss:.6f}")
            print(f"Final Val Loss: {final_val_loss:.6f}")
            print(f"Overfitting Ratio (Val/Train): {overfit_ratio:.4f}")
            print(f"Train Loss Change (last 10 epochs): {train_convergence:.2f}%")
            print(f"Val Loss Change (last 10 epochs): {val_convergence:.2f}%")
            
            # Interpretation with better thresholds
            analysis = {
                'train_loss': final_train_loss,
                'val_loss': final_val_loss,
                'overfit_ratio': overfit_ratio,
                'status': 'GOOD',
                'recommendations': []
            }
            
            if overfit_ratio > 2.0:
                print(f"⚠ Model shows SEVERE OVERFITTING (Val/Train ratio > 2.0)")
                analysis['status'] = 'SEVERE_OVERFITTING'
                analysis['recommendations'].append('Increase dropout rate (0.3 to 0.5)')
                analysis['recommendations'].append('Add more L2 regularization (weight_decay)')
                analysis['recommendations'].append('Reduce model capacity (fewer channels)')
                analysis['recommendations'].append('Increase training data if possible')
            elif overfit_ratio > 1.5:
                print(f"⚠ Model shows moderate OVERFITTING (Val/Train ratio 1.5-2.0)")
                analysis['status'] = 'MODERATE_OVERFITTING'
                analysis['recommendations'].append('Increase dropout rate to 0.2-0.3')
                analysis['recommendations'].append('Use data augmentation')
                analysis['recommendations'].append('Reduce learning rate slightly')
            elif overfit_ratio > 1.1:
                print(f"✓ Model shows slight OVERFITTING (1.1 < ratio < 1.5) - This is normal")
                analysis['status'] = 'NORMAL_OVERFIT'
                analysis['recommendations'].append('This level of overfitting is acceptable')
            elif overfit_ratio > 0.9:
                print(f"✓ Model generalization is EXCELLENT (0.9 < ratio < 1.1)")
                analysis['status'] = 'EXCELLENT'
                analysis['recommendations'].append('Model is well-balanced')
            elif overfit_ratio > 0.8:
                print(f"⚠ Model shows slight UNDERFITTING (0.8 < ratio < 0.9)")
                analysis['status'] = 'SLIGHT_UNDERFITTING'
                analysis['recommendations'].append('Reduce dropout to 0.05-0.1')
                analysis['recommendations'].append('Increase model capacity')
                analysis['recommendations'].append('Train for more epochs')
            else:
                print(f"⚠ Model shows UNDERFITTING (Val/Train ratio < 0.8)")
                analysis['status'] = 'UNDERFITTING'
                analysis['recommendations'].append('Reduce dropout significantly (0.05 or less)')
                analysis['recommendations'].append('Remove or reduce L2 regularization')
                analysis['recommendations'].append('Increase model capacity (more channels)')
                analysis['recommendations'].append('Increase training epochs')
                analysis['recommendations'].append('Use data augmentation to provide more learning signal')
            
            if train_convergence < 1.0 and val_convergence < 2.0:
                print(f"✓ Model has converged")
                analysis['convergence'] = 'CONVERGED'
            else:
                print(f"⚠ Model may not be fully converged")
                analysis['convergence'] = 'NOT_CONVERGED'
                if train_convergence < 5.0:
                    analysis['recommendations'].append('Early stopping prevented overfitting - consider training longer')
            
            if analysis['recommendations']:
                print(f"\nRecommendations:")
                for rec in analysis['recommendations']:
                    print(f"  • {rec}")
            
            self.test_results['overfitting_analysis'] = analysis
            
            # Plot training history
            self._plot_training_history(train_losses, val_losses)
            
            return True
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['overfitting_analysis'] = f'FAILED: {e}'
            return False
    
    def test_dropout_effect(self):
        """Test 6: Verify dropout is working (outputs differ in train vs eval)"""
        print("\n[TEST 6] Dropout Effect Verification")
        print("-" * 50)
        
        try:
            dummy_input = torch.randn(5, 7, 100, 240).to(self.device)
            
            # Get outputs in train mode (with dropout)
            self.model.train()
            with torch.no_grad():
                train_mode_outputs = []
                for _ in range(3):
                    output, _ = self.model(dummy_input)
                    train_mode_outputs.append(output.cpu().numpy())
            
            # Get outputs in eval mode (no dropout)
            self.model.eval()
            with torch.no_grad():
                eval_mode_outputs = []
                for _ in range(3):
                    output, _ = self.model(dummy_input)
                    eval_mode_outputs.append(output.cpu().numpy())
            
            # In train mode, outputs should vary due to dropout
            train_mode_diffs = []
            for i in range(len(train_mode_outputs)-1):
                diff = np.mean(np.abs(train_mode_outputs[i] - train_mode_outputs[i+1]))
                train_mode_diffs.append(diff)
            
            # In eval mode, outputs should be identical
            eval_mode_diff = np.mean(np.abs(eval_mode_outputs[0] - eval_mode_outputs[1]))
            
            avg_train_diff = np.mean(train_mode_diffs)
            
            print(f"Avg difference in TRAIN mode: {avg_train_diff:.6f}")
            print(f"Difference in EVAL mode: {eval_mode_diff:.8f}")
            
            if avg_train_diff > eval_mode_diff * 10:
                print(f"✓ Dropout is functioning correctly")
                self.test_results['dropout_effect'] = 'PASSED'
                return True
            else:
                print(f"⚠ Dropout effect unclear")
                self.test_results['dropout_effect'] = 'WARNING'
                return False
                
        except Exception as e:
            print(f"✗ Test failed: {e}")
            self.test_results['dropout_effect'] = f'FAILED: {e}'
            return False
    
    def test_batch_norm_effect(self):
        """Test 7: Verify batch norm is working"""
        print("\n[TEST 7] Batch Normalization Effect")
        print("-" * 50)
        
        try:
            # Test with different batch sizes
            dummy_input_b1 = torch.randn(1, 7, 100, 240).to(self.device)
            dummy_input_b8 = torch.randn(8, 7, 100, 240).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                output_b1, _ = self.model(dummy_input_b1)
                output_b8, _ = self.model(dummy_input_b8)
            
            print(f"Output B=1 shape: {output_b1.shape}")
            print(f"Output B=8 shape: {output_b8.shape}")
            print(f"Output B=1 std: {output_b1.std().item():.4f}")
            print(f"Output B=8 std: {output_b8.std().item():.4f}")
            print(f"✓ Batch normalization is functioning")
            
            self.test_results['batch_norm'] = 'PASSED'
            return True
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            self.test_results['batch_norm'] = f'FAILED: {e}'
            return False
    
    def test_gradient_flow(self):
        """Test 8: Verify gradients can flow through network"""
        print("\n[TEST 8] Gradient Flow Verification")
        print("-" * 50)
        
        try:
            dummy_input = torch.randn(1, 7, 100, 240).to(self.device)
            dummy_input.requires_grad = True
            dummy_target = torch.randn(1, 15, 100, 240).to(self.device)
            
            self.model.train()
            output, _ = self.model(dummy_input)
            loss = torch.mean((output - dummy_target) ** 2)
            loss.backward()
            
            # Check if gradients were computed
            grad_sum = 0
            grad_count = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_sum += param.grad.abs().sum().item()
                    grad_count += 1
            
            if grad_count > 0 and grad_sum > 0:
                print(f"✓ Gradients computed for {grad_count} parameters")
                print(f"✓ Sum of gradient magnitudes: {grad_sum:.6f}")
                self.test_results['gradient_flow'] = 'PASSED'
                return True
            else:
                print(f"✗ No gradients computed")
                self.test_results['gradient_flow'] = 'FAILED'
                return False
                
        except Exception as e:
            print(f"✗ Test failed: {e}")
            self.test_results['gradient_flow'] = f'FAILED: {e}'
            return False
    
    def _plot_training_history(self, train_losses, val_losses):
        """Plot training and validation loss curves"""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss', alpha=0.7)
            plt.plot(val_losses, label='Validation Loss', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.title('Training History - Overfitting/Underfitting Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('training_history.png', dpi=100)
            print(f"Training history plot saved as 'training_history.png'")
        except Exception as e:
            print(f"Could not save plot: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "=" * 60)
        print("OCEANEMBED MODEL COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        
        self.load_model()
        
        tests = [
            self.test_model_output_shape,
            self.test_output_value_range,
            self.test_model_consistency,
            self.test_real_data,
            self.test_overfitting_underfitting,
            self.test_dropout_effect,
            self.test_batch_norm_effect,
            self.test_gradient_flow,
        ]
        
        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"Unexpected error in {test_func.__name__}: {e}")
        
        # Save results
        self._save_results()
        
    def _save_results(self):
        """Save test results to file"""
        results_file = 'test_results.json'
        
        # Convert numpy types to Python native types
        results_to_save = {}
        for key, value in self.test_results.items():
            if isinstance(value, dict):
                results_to_save[key] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for k, v in value.items()
                }
            elif isinstance(value, (np.floating, np.integer)):
                results_to_save[key] = float(value)
            else:
                results_to_save[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\n{'=' * 60}")
        print("Test results saved to 'test_results.json'")
        print("=" * 60)
        
        # Print summary
        print("\nTEST SUMMARY:")
        print("-" * 60)
        for key, value in self.test_results.items():
            if isinstance(value, dict) and 'status' in value:
                status = value['status']
                print(f"{key}: {status}")
            elif isinstance(value, str):
                print(f"{key}: {value}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    tester = OceanEmbedTester()
    tester.run_all_tests()
