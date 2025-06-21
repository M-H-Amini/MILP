"""
Variational Autoencoder for image reconstruction error metric computation.
This module implements a lightweight convolutional VAE for computing reconstruction errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm


class ConvVAE(nn.Module):
    """
    Lightweight Convolutional Variational Autoencoder for 32x32 images.
    """
    
    def __init__(self, input_channels=3, latent_dim=64):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 4x4 -> 2x2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 2 * 2)
        
        # Decoder
        self.decoder = nn.Sequential(
            # 2x2 -> 4x4
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent space parameters."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent space to image."""
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 2, 2)
        return self.decoder(h)
    
    def forward(self, x):
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class ImageDataset(Dataset):
    """Dataset class for loading images from a directory."""
    
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function combining reconstruction loss and KL divergence.
    
    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term
    """
    # Reconstruction loss (binary cross entropy)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss


def train_vae(data_folder, model_name, epochs=50, batch_size=32, learning_rate=1e-3, device=None):
    """
    Train VAE on images from the specified folder with train/test split.
    
    Args:
        data_folder: Path to folder containing training images
        model_name: Name for saving the trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        device: Device to train on (cuda/cpu)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training VAE on device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    # Dataset and train/test split
    full_dataset = ImageDataset(data_folder, transform=transform)
    
    # Split dataset into train (80%) and test (20%)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Training on {len(train_dataset)} images, testing on {len(test_dataset)} images")
    
    # Model, optimizer
    model = ConvVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Variables to track best model
    best_test_loss = float('inf')
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for batch_idx, data in enumerate(train_pbar):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_dataset)
        
        # Testing phase
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            test_pbar = tqdm(test_dataloader, desc=f'Epoch {epoch+1}/{epochs} [Test]')
            for data in test_pbar:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                loss = vae_loss(recon_batch, data, mu, logvar)
                total_test_loss += loss.item()
                test_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_test_loss = total_test_loss / len(test_dataset)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Test Loss: {avg_test_loss:.4f}')
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict().copy()
            print(f'  New best model! Test Loss: {best_test_loss:.4f}')
    
    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, f'{model_name}.pth')
        print(f"Best model saved as {model_name}.pth with test loss: {best_test_loss:.4f}")
        
        # Load the best model state back into the model
        model.load_state_dict(best_model_state)
    else:
        # Fallback: save the final model
        torch.save(model.state_dict(), f'{model_name}.pth')
        print(f"Model saved as {model_name}.pth")
    
    return model


def load_vae(model_path, device=None):
    """Load a trained VAE model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ConvVAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def compute_vae_reconstruction_error(model, image1, image2, device=None):
    """
    Compute VAE reconstruction error between two images.
    
    Args:
        model: Trained VAE model
        image1: First image (PIL Image or tensor)
        image2: Second image (PIL Image or tensor)
        device: Device to compute on
    
    Returns:
        Reconstruction error as a scalar value
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    # Convert images to tensors if needed
    if isinstance(image1, Image.Image):
        image1 = transform(image1)
    if isinstance(image2, Image.Image):
        image2 = transform(image2)
    
    # Add batch dimension
    if len(image1.shape) == 3:
        image1 = image1.unsqueeze(0)
    if len(image2.shape) == 3:
        image2 = image2.unsqueeze(0)
    
    image1 = image1.to(device)
    image2 = image2.to(device)
    
    with torch.no_grad():
        # Reconstruct images
        recon1, _, _ = model(image1)
        recon2, _, _ = model(image2)
        
        # Compute reconstruction errors
        error1 = F.mse_loss(recon1, image1).item()
        error2 = F.mse_loss(recon2, image2).item()
        
        # Return average reconstruction error
        return (error1 + error2) / 2.0


if __name__ == "__main__":
    # Example usage
    data_folder = "Pair_Hu_Cifar10/original"
    model_name = "mh_vae_cifar10"
    
    # Train VAE
    model = train_vae(data_folder, model_name, epochs=100)
    print("VAE training completed!")