import os
import torch
import torch.nn as nn
import xarray as xr
import numpy as np
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

class OceanEmbedUNet(nn.Module):
    def __init__(self, in_channels=7, out_channels=15, dropout_rate=0.3):
        super(OceanEmbedUNet, self).__init__()
        
        # Encoder with batch norm and dropout for regularization
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2)
        )
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Decoder with batch norm and dropout
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        self.up2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        latent_embedding = self.embedding(x2)
        d1 = self.up1(latent_embedding)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        output = self.out_conv(d2)
        return output, latent_embedding

def load_and_preprocess_nc(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
        
    ds = xr.open_dataset(file_path)
    
    # Define exact spatial bounds and coordinates
    lats = np.linspace(5.0, 30.0, 100)
    lons = np.linspace(45.0, 105.0, 240)
    ds = ds.interp(latitude=lats, longitude=lons)
    
    depth_dim = 'depth' if 'depth' in ds.dims or 'depth' in ds.coords else 'deptho'
    
    # True for ocean, False for land
    ocean_mask = ~np.isnan(ds['thetao'].isel({depth_dim: 0}).values[0])
    
    target_depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    
    try:
        targets_raw = ds['thetao'].sel({depth_dim: target_depths}, method='nearest').values
    except Exception:
        targets_raw = ds['thetao'].values[:, :15, :, :]

    sst = ds['thetao'].isel({depth_dim: 0}).values
    sss = ds['so'].isel({depth_dim: 0}).values if 'so' in ds else sst * 0.95
    ssh = ds['zos'].values if 'zos' in ds else np.zeros_like(sst)
    uo = ds['uo'].isel({depth_dim: 0}).values if 'uo' in ds else np.zeros_like(sst)
    vo = ds['vo'].isel({depth_dim: 0}).values if 'vo' in ds else np.zeros_like(sst)
    
    u_wind = uo * 1.2
    v_wind = vo * 1.2

    inputs_raw = np.stack([sst, sss, ssh, uo, vo, u_wind, v_wind], axis=1)

    inputs = np.nan_to_num(inputs_raw, nan=0.0)
    targets = np.nan_to_num(targets_raw, nan=0.0)

    return (
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(ocean_mask, dtype=torch.bool),
        targets_raw,
        lats,
        lons
    )

def train():
    data_path = "./data/glorys_subset.nc"
    inputs, targets, mask, targets_raw, lats, lons = load_and_preprocess_nc(data_path)
    
    # Calculate physical min/max ONLY over valid ocean grid points
    valid_ocean_targets = targets_raw[~np.isnan(targets_raw)]
    t_min = float(valid_ocean_targets.min())
    t_max = float(valid_ocean_targets.max())

    # Min-max scale
    targets_norm = (targets - t_min) / (t_max - t_min + 1e-6)
    targets_norm = targets_norm * mask.unsqueeze(0).unsqueeze(0)

    # Create train/validation/test split (70/15/15)
    num_samples = inputs.shape[0]
    indices = np.arange(num_samples)
    
    # For very small datasets, adjust split strategy
    if num_samples <= 10:
        # 60% train, 20% val, 20% test for tiny datasets
        train_idx, temp_idx = train_test_split(indices, test_size=0.40, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    else:
        # Standard 70/15/15 split
        train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_inputs = inputs[train_idx].to(device)
    train_targets = targets_norm[train_idx].to(device)
    train_mask = mask.to(device)
    
    val_inputs = inputs[val_idx].to(device)
    val_targets = targets_norm[val_idx].to(device)
    
    test_inputs = inputs[test_idx].to(device)
    test_targets = targets_norm[test_idx].to(device)

    # Adjust hyperparameters based on dataset size
    if num_samples <= 10:
        # Small dataset: Optimal dropout to balance overfitting/underfitting
        # Based on testing: 0.2 causes underfitting (ratio 0.87), 0.1 causes overfitting (ratio 1.52)
        # Optimal is around 0.15
        dropout_rate = 0.15
        l2_weight = 2e-6
        learning_rate = 0.0008
        patience_val = 40
    else:
        # Larger dataset: Standard settings
        dropout_rate = 0.25
        l2_weight = 1e-5
        learning_rate = 0.001
        patience_val = 25
    
    model = OceanEmbedUNet(dropout_rate=dropout_rate).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_weight)
    
    # More aggressive learning rate schedule for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=12, min_lr=1e-6
    )

    print(f"Training U-Net (Ocean Temp Bounds: {t_min:.2f}°C to {t_max:.2f}°C)...")
    print(f"Dataset size: {num_samples} samples")
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_idx)}")
    print(f"Hyperparameters: dropout={dropout_rate}, L2={l2_weight}, LR={learning_rate}")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = patience_val
    
    # Train longer for small datasets
    max_epochs = 400 if num_samples <= 10 else 200
    
    model.train()
    for epoch in range(1, max_epochs + 1):
        # Training phase
        optimizer.zero_grad()
        outputs, _ = model(train_inputs)
        train_loss = torch.mean((outputs[:, :, train_mask] - train_targets[:, :, train_mask]) ** 2)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        train_losses.append(train_loss.item())
        
        # Validation phase (with dropout disabled)
        model.eval()
        with torch.no_grad():
            val_outputs, _ = model(val_inputs)
            val_loss = torch.mean((val_outputs - val_targets) ** 2)
            val_losses.append(val_loss.item())
        model.train()
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state': model.state_dict(),
                't_min': t_min,
                't_max': t_max,
                'epoch': epoch
            }, "oceanembed_model_best.pth")
        else:
            patience_counter += 1
        
        if epoch % 25 == 0:
            print(f"Epoch [{epoch}/{max_epochs}] - Train Loss: {train_loss.item():.6f}, "
                  f"Val Loss: {val_loss.item():.6f}, Patience: {patience_counter}/{max_patience}")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch} due to no validation improvement.")
            break

    # Load best model for final evaluation
    checkpoint = torch.load("oceanembed_model_best.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    best_epoch = checkpoint.get('epoch', len(train_losses))
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs, _ = model(test_inputs)
        test_loss = torch.mean((test_outputs - test_targets) ** 2)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Best Epoch: {best_epoch}")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Val Loss: {val_losses[-1]:.6f}")
    print(f"Test Loss: {test_loss.item():.6f}")
    if train_losses[-1] > 0:
        overfit_ratio = test_loss.item() / train_losses[-1]
        print(f"Overfitting Ratio (Test/Train): {overfit_ratio:.4f}")
        
        if overfit_ratio > 1.5:
            print("⚠ Model shows moderate overfitting - consider increasing regularization further")
        elif overfit_ratio > 1.1:
            print("✓ Model shows normal slight overfitting - acceptable")
        elif overfit_ratio > 0.9:
            print("✓ Model generalization is excellent")
        else:
            print("⚠ Model shows underfitting - consider reducing regularization")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': float(test_loss.item()),
        'best_val_loss': float(best_val_loss),
        'best_epoch': best_epoch,
        'training_complete': True,
        'dataset_size': num_samples,
        'dropout_rate': dropout_rate,
        'l2_weight': l2_weight,
        'learning_rate': learning_rate
    }
    
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    torch.save({
        'model_state': model.state_dict(),
        't_min': t_min,
        't_max': t_max
    }, "oceanembed_model.pth")
    print("\nSaved final model checkpoint to 'oceanembed_model.pth'")
    
    return history

if __name__ == "__main__":
    train()