"""
Image Metrics Computation Module

This module computes 13 different image comparison metrics for pairs of images:
1. PSNR (Peak Signal-to-Noise Ratio)
2. MSE (Mean Squared Error)
3. SSIM (Structural Similarity Index)
4. TSI (Textual Similarity Index)
5. WS (Wasserstein Score)
6. CS (Cosine Similarity based on VGG latent space)
7. KL (Kullback-Leibler divergence)
8. HistI (Histogram Intersection)
9. HistC (Histogram Correlation)
10. CPL (Classifier Perceptual Loss using VGG)
11. SSS (Semantic Segmentation Score)
12. VAE-RE (Variational Autoencoder Reconstruction Error)
13. VIF (Visual Information Fidelity)
"""

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Scientific computing imports
from scipy import ndimage
from scipy.stats import wasserstein_distance, entropy
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics.pairwise import cosine_similarity

# Import VAE module
from mh_vae import train_vae, load_vae, compute_vae_reconstruction_error, ConvVAE

# Tqdm for progress bars
from tqdm import tqdm


class ImageMetricsCalculator:
    """
    A comprehensive image metrics calculator that computes various similarity/distance metrics 
    between pairs of images.
    """
    
    def __init__(self, device=None):
        """Initialize the metrics calculator with required models."""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Initializing ImageMetricsCalculator on device: {self.device}")
        
        # Initialize VGG model for feature extraction
        self.vgg_model = self._load_vgg_model()
        
        # Initialize segmentation model (using DeepLabV3)
        self.segmentation_model = self._load_segmentation_model()
        
        # Transform for preprocessing images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transform for segmentation
        self.seg_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # VAE model placeholder
        self.vae_model = None
    
    def _load_vgg_model(self):
        """Load pre-trained VGG model for feature extraction."""
        vgg = models.vgg16(pretrained=True)
        # Remove the classifier and keep only features
        vgg = nn.Sequential(*list(vgg.features.children()))
        vgg.eval()
        return vgg.to(self.device)
    
    def _load_segmentation_model(self):
        """Load pre-trained segmentation model."""
        model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        model.eval()
        return model.to(self.device)
    
    def load_vae_model(self, model_path):
        """Load the trained VAE model."""
        if os.path.exists(model_path):
            self.vae_model = load_vae(model_path, self.device)
            print(f"VAE model loaded from {model_path}")
        else:
            print(f"VAE model not found at {model_path}")
    
    def _preprocess_image(self, image_path, target_size=(224, 224)):
        """Load and preprocess image."""
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image.resize(target_size))
        return image, image_array
    
    def compute_psnr(self, img1_path: str, img2_path: str) -> float:
        """Compute Peak Signal-to-Noise Ratio."""
        _, img1 = self._preprocess_image(img1_path)
        _, img2 = self._preprocess_image(img2_path)
        
        try:
            return psnr(img1, img2)
        except:
            return 0.0
    
    def compute_mse(self, img1_path: str, img2_path: str) -> float:
        """Compute Mean Squared Error."""
        _, img1 = self._preprocess_image(img1_path)
        _, img2 = self._preprocess_image(img2_path)
        
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        return float(mse)
    
    def compute_ssim(self, img1_path: str, img2_path: str) -> float:
        """Compute Structural Similarity Index."""
        _, img1 = self._preprocess_image(img1_path)
        _, img2 = self._preprocess_image(img2_path)
        
        try:
            # Convert to grayscale for SSIM
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            return ssim(img1_gray, img2_gray)
        except:
            return 0.0
    
    def compute_tsi(self, img1_path: str, img2_path: str) -> float:
        """
        Compute Textual Similarity Index using edge detection.
        This is a custom metric based on edge similarity.
        """
        _, img1 = self._preprocess_image(img1_path)
        _, img2 = self._preprocess_image(img2_path)
        
        # Convert to grayscale and apply edge detection
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        edges1 = cv2.Canny(img1_gray, 50, 150)
        edges2 = cv2.Canny(img2_gray, 50, 150)
        
        # Compute correlation between edge maps
        correlation = np.corrcoef(edges1.flatten(), edges2.flatten())[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def compute_wasserstein_score(self, img1_path: str, img2_path: str) -> float:
        """Compute Wasserstein distance between image histograms."""
        _, img1 = self._preprocess_image(img1_path)
        _, img2 = self._preprocess_image(img2_path)
        
        # Convert to grayscale and compute histograms
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256]).flatten()
        hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256]).flatten()
        
        # Normalize histograms
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        try:
            return wasserstein_distance(np.arange(256), np.arange(256), hist1, hist2)
        except:
            return 0.0
    
    def compute_cosine_similarity_vgg(self, img1_path: str, img2_path: str) -> float:
        """Compute cosine similarity using VGG features."""
        img1, _ = self._preprocess_image(img1_path)
        img2, _ = self._preprocess_image(img2_path)
        
        # Transform images
        img1_tensor = self.transform(img1).unsqueeze(0).to(self.device)
        img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract features
            features1 = self.vgg_model(img1_tensor).view(1, -1).cpu().numpy()
            features2 = self.vgg_model(img2_tensor).view(1, -1).cpu().numpy()
            
            # Compute cosine similarity
            similarity = cosine_similarity(features1, features2)[0, 0]
            return float(similarity)
    
    def compute_kl_divergence(self, img1_path: str, img2_path: str) -> float:
        """Compute Kullback-Leibler divergence between image histograms."""
        _, img1 = self._preprocess_image(img1_path)
        _, img2 = self._preprocess_image(img2_path)
        
        # Convert to grayscale and compute histograms
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256]).flatten()
        hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256]).flatten()
        
        # Normalize and add small epsilon to avoid log(0)
        hist1 = hist1 / np.sum(hist1) + 1e-10
        hist2 = hist2 / np.sum(hist2) + 1e-10
        
        try:
            return entropy(hist1, hist2)
        except:
            return 0.0
    
    def compute_histogram_intersection(self, img1_path: str, img2_path: str) -> float:
        """Compute histogram intersection."""
        _, img1 = self._preprocess_image(img1_path)
        _, img2 = self._preprocess_image(img2_path)
        
        # Convert to grayscale and compute histograms
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256]).flatten()
        hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256]).flatten()
        
        # Normalize histograms
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        # Compute intersection
        intersection = np.sum(np.minimum(hist1, hist2))
        return float(intersection)
    
    def compute_histogram_correlation(self, img1_path: str, img2_path: str) -> float:
        """Compute histogram correlation."""
        _, img1 = self._preprocess_image(img1_path)
        _, img2 = self._preprocess_image(img2_path)
        
        # Convert to grayscale and compute histograms
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256]).flatten()
        hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256]).flatten()
        
        # Compute correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return float(correlation)
    
    def compute_classifier_perceptual_loss(self, img1_path: str, img2_path: str) -> float:
        """Compute classifier perceptual loss using VGG features."""
        img1, _ = self._preprocess_image(img1_path)
        img2, _ = self._preprocess_image(img2_path)
        
        # Transform images
        img1_tensor = self.transform(img1).unsqueeze(0).to(self.device)
        img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract features from multiple layers
            features1 = self.vgg_model(img1_tensor)
            features2 = self.vgg_model(img2_tensor)
            
            # Compute MSE between features
            mse_loss = F.mse_loss(features1, features2)
            return float(mse_loss.cpu().item())
    
    def compute_semantic_segmentation_score(self, img1_path: str, img2_path: str) -> float:
        """Compute semantic segmentation similarity score."""
        img1, _ = self._preprocess_image(img1_path, (512, 512))
        img2, _ = self._preprocess_image(img2_path, (512, 512))
        
        # Transform images for segmentation
        img1_tensor = self.seg_transform(img1).unsqueeze(0).to(self.device)
        img2_tensor = self.seg_transform(img2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get segmentation maps
            seg1 = self.segmentation_model(img1_tensor)['out']
            seg2 = self.segmentation_model(img2_tensor)['out']
            
            # Convert to class predictions
            seg1_pred = torch.argmax(seg1, dim=1).cpu().numpy()
            seg2_pred = torch.argmax(seg2, dim=1).cpu().numpy()
            
            # Compute IoU (Intersection over Union)
            intersection = np.sum(seg1_pred == seg2_pred)
            total_pixels = seg1_pred.size
            
            return float(intersection / total_pixels)
    
    def compute_vae_reconstruction_error(self, img1_path: str, img2_path: str) -> float:
        """Compute VAE reconstruction error."""
        if self.vae_model is None:
            return 0.0
        
        img1, _ = self._preprocess_image(img1_path, (32, 32))
        img2, _ = self._preprocess_image(img2_path, (32, 32))
        
        return compute_vae_reconstruction_error(self.vae_model, img1, img2, self.device)
    
    def compute_vif(self, img1_path: str, img2_path: str) -> float:
        """
        Compute Visual Information Fidelity.
        This is a simplified implementation using mutual information.
        """
        _, img1 = self._preprocess_image(img1_path)
        _, img2 = self._preprocess_image(img2_path)
        
        # Convert to grayscale
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Compute histograms
        hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256]).flatten()
        hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256]).flatten()
        
        # Normalize
        hist1 = hist1 / np.sum(hist1) + 1e-10
        hist2 = hist2 / np.sum(hist2) + 1e-10
        
        # Compute mutual information approximation
        mi = np.sum(hist1 * np.log2(hist1 / hist2))
        return float(abs(mi))
    
    def compute_all_metrics(self, img1_path: str, img2_path: str) -> List[float]:
        """Compute all 13 metrics for a pair of images."""
        metrics = []
        
        # 1. PSNR
        metrics.append(self.compute_psnr(img1_path, img2_path))
        
        # 2. MSE
        metrics.append(self.compute_mse(img1_path, img2_path))
        
        # 3. SSIM
        metrics.append(self.compute_ssim(img1_path, img2_path))
        
        # 4. TSI
        metrics.append(self.compute_tsi(img1_path, img2_path))
        
        # 5. WS (Wasserstein Score)
        metrics.append(self.compute_wasserstein_score(img1_path, img2_path))
        
        # 6. CS (Cosine Similarity)
        metrics.append(self.compute_cosine_similarity_vgg(img1_path, img2_path))
        
        # 7. KL (Kullback-Leibler)
        metrics.append(self.compute_kl_divergence(img1_path, img2_path))
        
        # 8. HistI (Histogram Intersection)
        metrics.append(self.compute_histogram_intersection(img1_path, img2_path))
        
        # 9. HistC (Histogram Correlation)
        metrics.append(self.compute_histogram_correlation(img1_path, img2_path))
        
        # 10. CPL (Classifier Perceptual Loss)
        metrics.append(self.compute_classifier_perceptual_loss(img1_path, img2_path))
        
        # 11. SSS (Semantic Segmentation Score)
        metrics.append(self.compute_semantic_segmentation_score(img1_path, img2_path))
        
        # 12. VAE-RE (VAE Reconstruction Error)
        metrics.append(self.compute_vae_reconstruction_error(img1_path, img2_path))
        
        # 13. VIF (Visual Information Fidelity)
        metrics.append(self.compute_vif(img1_path, img2_path))
        
        return metrics


def computeMetrics(ds_folder, input_csv_file='cifar10_prepared_dataset.csv', output_csv_file='cifar10_metrics.csv'):
    """
    Main function to compute metrics for all image pairs in the dataset.
    
    Args:
        ds_folder: Path to the folder containing image pairs (e.g., 'Pair_Hu_Cifar10')
        input_csv_file: CSV file containing image pair information
        output_csv_file: Output CSV file to save computed metrics
    
    Returns:
        Tuple of (X, y) where X is metrics array and y is labels array
    """
    
    print("Starting metrics computation...")
    
    # Read the CSV file
    df = pd.read_csv(input_csv_file)
    print(f"Processing {len(df)} image pairs")
    
    # Initialize metrics calculator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    calculator = ImageMetricsCalculator(device)
    
    # Check if VAE model exists, if not train it
    vae_model_path = 'mh_vae_cifar10.pth'
    if not os.path.exists(vae_model_path):
        print("VAE model not found. Training VAE...")
        original_folder = os.path.join(ds_folder, 'original')
        if os.path.exists(original_folder):
            from mh_vae import train_vae
            train_vae(original_folder, 'mh_vae_cifar10', epochs=30)
        else:
            print(f"Warning: Original folder not found at {original_folder}")
    
    # Load VAE model
    calculator.load_vae_model(vae_model_path)
    
    # Prepare data structures
    all_metrics = []
    labels = []
    
    # Column names for metrics
    metric_names = ['PSNR', 'MSE', 'SSIM', 'TSI', 'WS', 'CS', 'KL', 'HistI', 'HistC', 'CPL', 'SSS', 'VAE_RE', 'VIF']
    
    # Process each image pair
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing metrics"):
        try:
            # Extract image paths from URLs
            image_id = row['image_id']
            
            # Construct local file paths
            original_path = os.path.join(ds_folder, 'original', image_id.replace('/', '_'))
            transformed_path = os.path.join(ds_folder, 'transformed', image_id.replace('/', '_'))
            
            # Check if files exist
            if not os.path.exists(original_path) or not os.path.exists(transformed_path):
                print(f"Warning: Files not found for {image_id}")
                continue
            
            # Compute all metrics
            metrics = calculator.compute_all_metrics(original_path, transformed_path)
            all_metrics.append(metrics)
            
            # Extract label (0 for valid, 1 for invalid)
            labels.append(row['invalid'])
            
        except Exception as e:
            print(f"Error processing {image_id}: {str(e)}")
            continue
    
    # Convert to numpy arrays
    X = np.array(all_metrics)
    y = np.array(labels)
    
    print(f"Successfully computed metrics for {len(X)} image pairs")
    print(f"Metrics shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Create output DataFrame
    output_df = pd.DataFrame(X, columns=metric_names)
    output_df['label'] = y
    
    # Save to CSV
    output_df.to_csv(output_csv_file, index=False)
    print(f"Results saved to {output_csv_file}")
    
    return X, y


if __name__ == "__main__":
    # Example usage
    ds_folder = 'Pair_Hu_Cifar10'
    input_csv_file = 'cifar10_prepared_dataset.csv'
    output_csv_file = 'cifar10_metrics.csv'
    X, y = computeMetrics(ds_folder, input_csv_file, output_csv_file)
    print("Metrics computation completed!")
