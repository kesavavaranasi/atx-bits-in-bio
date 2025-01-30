import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import logging
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn

torch.backends.cudnn.benchmark = True  # Speed up fixed-size inputs
torch.set_float32_matmul_precision('medium')  # Faster matrix operations
                
logging.basicConfig(level = logging.INFO)

if torch.cuda.is_available():
    cudnn.benchmark = True  # Speed up training


def load_data(
    data_path: Path = Path("protein_localizations.h5"), 
    protein_id: str = None
):
    """Load data from H5 file or specific protein"""
    data = {}
    with h5py.File(data_path, 'r') as f:
        # If protein_id is specified, only load that protein's data
        if protein_id:
            for location in f.keys():
                if protein_id in f[location]:
                    location_data = {}
                    sample_group = f[location][protein_id]
                    
                    # Get all images for this sample
                    images = {}
                    for img_name in sample_group.keys():
                        image_data = sample_group[img_name][:]
                        image_id = sample_group[img_name].attrs['image_id']
                        
                        images[image_id] = {
                            'data': image_data,
                            'filename': sample_group[img_name].attrs['filename'],
                        }
                    
                    location_data[protein_id] = {
                        'images': images,
                        'num_images': sample_group.attrs['num_images'],
                        'date_processed': sample_group.attrs['date_processed']
                    }
                    data[location] = location_data
                    break
        # Otherwise load all data
        else:
            for location in f.keys():
                location_data = {}
                
                for sample_id in tqdm(f[location].keys(), desc=f"Loading {location} data"):
                    sample_group = f[location][sample_id]
                    
                    # Get all images for this sample
                    images = {}
                    for img_name in sample_group.keys():
                        image_data = sample_group[img_name][:]
                        image_id = sample_group[img_name].attrs['image_id']
                        
                        images[image_id] = {
                            'data': image_data,
                            'filename': sample_group[img_name].attrs['filename'],
                        }
                    
                    location_data[sample_id] = {
                        'images': images,
                        'num_images': sample_group.attrs['num_images'],
                        'date_processed': sample_group.attrs['date_processed']
                    }
                
                data[location] = location_data
    
    return data


class CellDataset(Dataset):
    """Dataset class for cell images with categorical labels"""
    def __init__(
        self, 
        data_dict: dict,
        img_size: tuple = (512, 512)
    ):
        self.img_size = img_size
        
        # Initialize lists
        self.images = []
        self.labels = []
        
        # Create label encoder
        self.label_encoder = LabelEncoder()
        categories = list(data_dict.keys())
        self.label_encoder.fit(categories)
        logging.info(f"Label mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        # Process images with robust normalization
        for category in categories:
            category_data = data_dict[category]
            encoded_label = self.label_encoder.transform([category])[0]
            
            for sample_id, sample_data in category_data.items():
                for image_id, image_info in sample_data['images'].items():
                    # Get image data (2 channels: nuclear and protein)
                    img = image_info['data'].astype(np.float32)
                    
                    # Normalize each channel independently
                    normalized_img = np.zeros_like(img)
                    for c in range(img.shape[0]):
                        # Get non-background pixels
                        signal = img[c][img[c] > 0]
                        if len(signal) > 0:
                            # Use robust statistics (98th percentile instead of max)
                            p98 = np.percentile(signal, 98)
                            # Normalize by 98th percentile (common in fluorescence microscopy)
                            normalized_img[c] = np.clip(img[c], 0, p98) / p98
                    
                    # Convert to tensor
                    img_tensor = torch.from_numpy(normalized_img)
                    
                    # Resize if needed
                    if img_tensor.shape[-2:] != self.img_size:
                        img_tensor = F.interpolate(
                            img_tensor.unsqueeze(0), 
                            size=self.img_size, 
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                    
                    self.images.append(img_tensor)
                    self.labels.append(encoded_label)
        
        self.images = torch.stack(self.images)
        self.labels = torch.LongTensor(self.labels)
        
        logging.info(f"Dataset created with {len(self.images)} images")

    def normalize_microscopy_image(self, image):
        """Normalize microscopy image using robust statistics"""
        # Convert to float32 for calculations
        image = image.astype(np.float32)
        
        # Calculate robust statistics per channel
        for c in range(image.shape[0]):
            # Get non-zero pixels (background is usually 0 in microscopy)
            non_zero = image[c][image[c] > 0]
            if len(non_zero) > 0:
                # Use percentiles instead of min/max to handle outliers
                p1, p99 = np.percentile(non_zero, [1, 99.9])
                
                # Clip extreme values
                image[c] = np.clip(image[c], p1, p99)
                
                # Normalize to [0, 1] range
                image[c] = (image[c] - p1) / (p99 - p1 + 1e-8)
        
        return image
    
    def log_transform(self, image):
        """Apply log transform to handle large intensity ranges"""
        # Add small constant to avoid log(0)
        return np.log1p(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class DualStreamProteinClassifier(nn.Module):
    """
    Dual-stream CNN for protein localization classification
    - Separate initial processing for nuclear and protein channels
    - Early fusion to allow nuclear context to guide protein feature extraction
    - Lightweight architecture for fast training
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # Instance normalization instead of batch norm for more stable training
        self.input_norm = nn.InstanceNorm2d(2, affine=True)
        
        # Simplified nuclear stream - reduced channels and depth
        self.nuclear_stream = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=4, padding=2, bias=False),  # Larger stride, fewer channels
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.Dropout(0.1),
        )
        
        # Simplified protein stream
        self.protein_stream = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.Dropout(0.1),
        )
      
        # Lightweight attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(96, 96, 1, bias=False),
            nn.Sigmoid()
        )
    
    
        # Efficient classifier
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, num_classes),
            nn.Dropout(0.1),
        )
        

    # @torch.compile  # JIT compilation for faster inference - commented out due to Windows compiler issues
    def forward(self, x):
        x = self.input_norm(x)
    
        # Parallel processing of streams
        n = self.nuclear_stream(x[:, 0:1])
        p = self.protein_stream(x[:, 1:2])
   
        # Combine features
        combined = torch.cat([n, p], dim=1)    
        combined = combined * self.channel_attention(combined)
        return self.fusion(combined)


def model_inference(model, protein_id, device='cuda', dataset=None):
    """Run inference on a single protein"""
    try:
        if dataset is None:
            raise ValueError("Dataset must be provided for label encoder mapping")
            
        # Get the label mapping from the training dataset
        label_mapping = dict(zip(
            range(len(dataset.label_encoder.classes_)), 
            dataset.label_encoder.classes_
        ))
        logging.info(f"Available labels: {label_mapping}")
        
        # Load the protein data
        data = load_data(protein_id = protein_id)
        
        # Find the protein in any category
        protein_data = None
        for category_data in data.values():
            if protein_id in category_data:
                protein_data = category_data[protein_id]
                break
        
        if protein_data is None:
            raise ValueError(f"Protein {protein_id} not found in dataset")
            
        # Create temporary dataset using the ORIGINAL label encoder
        temp_dataset = CellDataset({'temp_category': {protein_id: protein_data}})
        
        # Get the first image
        image = temp_dataset.images[0].unsqueeze(0).to(device)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1)
            pred_class = outputs.argmax(1).item()
            
            # Use the original label mapping
            pred_label = label_mapping[pred_class]
            
            # Get probability distribution
            class_probs = {
                label_mapping[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
            
            return {
                'predicted_class': pred_label,
                'probabilities': class_probs
            }
            
    except Exception as e:
        logging.error(f"Inference failed for protein {protein_id}: {str(e)}")
        raise


def create_train_test_split(data_dict: dict, test_size: float = 0.2):
    """Split data into train and test sets"""
    train_data = {}
    test_data = {}
    
    for category, samples in data_dict.items():
        # Calculate split point
        n_samples = len(samples)
        n_test = max(1, int(n_samples * test_size))
        
        # Randomly select test samples
        all_sample_ids = list(samples.keys())
        np.random.shuffle(all_sample_ids)
        
        test_samples = all_sample_ids[:n_test]
        train_samples = all_sample_ids[n_test:]
        
        # Split the data
        train_data[category] = {
            sample_id: samples[sample_id] 
            for sample_id in train_samples
        }
        
        test_data[category] = {
            sample_id: samples[sample_id]
            for sample_id in test_samples
        }
    
    return train_data, test_data


def train_model(model, train_loader, test_loader, num_epochs=25, device='cuda'):
    """Training loop with test set monitoring"""
    try:
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=5e-3,
            weight_decay=0.01,
            eps=1e-8
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-3,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.2,
            div_factor=5,
            final_div_factor=50
        )
        
        scaler = GradScaler()
        
        # Metrics storage
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        
        best_test_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss, train_acc = _train_epoch(
                model, train_loader, optimizer, criterion, 
                scaler, scheduler, device
            )
            
            # Testing phase
            model.eval()
            test_loss, test_acc = _evaluate(model, test_loader, criterion, device)
            
            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            
            # Log metrics
            logging.info(f'Epoch {epoch+1}/{num_epochs}:')
            logging.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logging.info(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            
            # Save checkpoint if test loss improved
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'train_acc': train_acc,
                    'test_acc': test_acc
                }, 'best_model.pth')
            
            # Visualizations
            if (epoch + 1) % 2 == 0:
                plot_training_curves(
                    train_losses, train_accuracies,
                    test_losses, test_accuracies,
                    Path('outputs'), epoch
                )
        
        return model
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise


def _train_epoch(model, loader, optimizer, criterion, scaler, scheduler, device):
    """Run one epoch of training"""
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc='Training')
    
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f'{running_loss/(progress_bar.n+1):.3f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    return running_loss / len(loader), 100. * correct / total


def _evaluate(model, loader, criterion, device):
    """Evaluate model on the provided loader"""
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    
    return running_loss / len(loader), 100. * correct / total


def plot_training_curves(train_losses, train_accuracies, test_losses, test_accuracies, output_dir, epoch):
    """Plot and save training curves with test metrics"""
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(121)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(test_losses, 'r-', label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot accuracies
    plt.subplot(122)
    plt.plot(train_accuracies, 'b-', label='Train Accuracy')
    plt.plot(test_accuracies, 'r-', label='Test Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_epoch_{epoch+1}.png')
    plt.close()


if __name__ == "__main__":
    try:
        # Speed optimizations
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('medium')
        
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        # Load and split data
        data = load_data()
        train_data, test_data = create_train_test_split(data)
        
        # Create datasets
        train_dataset = CellDataset(train_data)
        test_dataset = CellDataset(test_data)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size = 16,
            shuffle = True,
            num_workers = 4,
            pin_memory = True,
            persistent_workers = True,
            prefetch_factor = 2,
            drop_last = True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size = 16,
            shuffle = False,
            num_workers = 4,
            pin_memory = True,
            persistent_workers = True,
            prefetch_factor = 2
        )
        
        # Create and train model
        num_classes = len(train_dataset.label_encoder.classes_)
        model = DualStreamProteinClassifier(num_classes)
        
        trained_model = train_model(
            model,
            train_loader,
            test_loader,
            num_epochs = 25,
            device = device
        )
        
        results = model_inference(
            trained_model, 
            'ENSG00000184743', 
            device = device,
            dataset = train_dataset  # Pass the training dataset
        )
        logging.info(f"Inference results: {results}")
        
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
        raise
