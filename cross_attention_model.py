import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
from datetime import datetime
import logging
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
from collections import defaultdict
from adabelief_pytorch import AdaBelief
 
torch.backends.cudnn.benchmark = True  # Speed up fixed-size inputs
torch.set_float32_matmul_precision('medium')  # Faster matrix operations
                
logging.basicConfig(level = logging.INFO)

if torch.cuda.is_available():
    cudnn.benchmark = True  # Speed up training


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        squeeze = self.squeeze(x).view(b, c)
        excite = self.excite(squeeze).view(b, c, 1, 1)
        return x * excite.expand_as(x)


class CrossAttention(nn.Module):
    """Memory-efficient cross-attention"""
    def __init__(self, in_channels, out_channels, reduction=8):
        super().__init__()
        self.query = nn.Conv2d(in_channels, out_channels // reduction, 1)
        self.key = nn.Conv2d(in_channels, out_channels // reduction, 1)
        self.value = nn.Conv2d(in_channels, out_channels, 1)
        self.scale = (out_channels // reduction) ** -0.5
        
    def forward(self, x, context):
        # Compute attention scores efficiently
        q = self.query(x)
        k = self.key(context)
        v = self.value(context)
        
        # Reshape for attention
        b, c, h, w = q.shape
        q = q.view(b, c, -1)
        k = k.view(b, c, -1)
        v = v.view(b, v.size(1), -1)
        
        # Efficient attention
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.bmm(attn, v.transpose(1, 2))
        return out.view(b, v.size(1), h, w)


class ResidualBlock(nn.Module):
    """Pre-activation residual block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.silu = nn.SiLU(inplace=True)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = self.silu(self.norm1(x))
        out = self.conv1(out)
        out = self.silu(self.norm2(out))
        out = self.conv2(out)
        return out + self.shortcut(x)


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
                    img = torch.from_numpy(image_info['data'].astype(np.float32))
                    
                    # Normalize each channel independently
                    normalized_img = torch.zeros_like(img)
                    for c in range(img.shape[0]):
                        channel_data = img[c]
                        
                        # Get non-background pixels
                        signal = channel_data[channel_data > 0]
                        
                        if len(signal) > 0:
                            # Use robust statistics (98th percentile instead of max)
                            p98 = torch.quantile(signal, 0.98)
                            # Normalize by 98th percentile (common in fluorescence microscopy)
                            channel_norm = torch.clamp(channel_data, 0, p98) / p98
                            normalized_img[c] = channel_norm
                    
                    # Resize if needed
                    if normalized_img.shape[-2:] != self.img_size:
                        normalized_img = F.interpolate(
                            normalized_img.unsqueeze(0), 
                            size=self.img_size, 
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                    
                    self.images.append(normalized_img)
                    self.labels.append(encoded_label)
        
        self.images = torch.stack(self.images)
        self.labels = torch.LongTensor(self.labels)
        
        logging.info(f"Dataset created with {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class DualStreamProteinClassifier(nn.Module):
    """Dual-stream CNN for protein localization with efficient cross-attention"""
    def __init__(self, num_classes):
        super().__init__()
        
        # Input normalization - changed from 8 groups to 2 for 2 channels
        self.input_norm = nn.GroupNorm(2, 2)  # One group per channel for input
        
        # Shared initial processing
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 32),  
            nn.SiLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Efficient feature extraction streams
        self.stream_blocks = nn.ModuleList([
            # Block 1
            nn.Sequential(
                ResidualBlock(32, 64, stride=2),
                SEBlock(64),
            ),
            # Block 2 
            nn.Sequential(
                ResidualBlock(64, 128, stride=2),
                SEBlock(128),
            ),
            # Block 3
            nn.Sequential(
                ResidualBlock(128, 256),
                SEBlock(256),
            )
        ])
        
        # Efficient cross-attention
        self.cross_attention = CrossAttention(256, 256, reduction=8)
        
        # Fast global context
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1)  # Reduced dropout for faster convergence
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Split channels
        nuclear = x[:, 0:1]
        protein = x[:, 1:2]
        
        # Initial shared processing
        n = self.stem(nuclear)
        p = self.stem(protein)
        
        # Progressive feature extraction
        for block in self.stream_blocks:
            n = block(n)
            p = block(p)
        
        # Cross-attention fusion
        attended_p = self.cross_attention(p, n)
        attended_n = self.cross_attention(n, p)
        
        # Combine features
        fused = torch.cat([attended_n, attended_p], dim=1)
        
        # Global context and classification
        out = self.global_context(fused)
        return self.classifier(out)


def model_inference(model, protein_id: str, device='cuda'):
    """Run inference on a single protein's images"""
    try:
        # Create temporary data structure matching training format
        image_dir = Path(f"opencell_data/images/{protein_id}")
        if not image_dir.exists():
            raise ValueError(f"No images found for protein {protein_id}")
            
        # Load and preprocess image
        image_files = list(image_dir.glob("*.tif"))
        if not image_files:
            raise ValueError(f"No .tif files found in {image_dir}")
            
        # Load first image and structure like training data
        image_data = tifffile.imread(str(image_files[0]))
        temp_data = {
            'temp_category': {
                protein_id: {
                    'images': {
                        'img_0': {
                            'data': image_data,
                            'filename': image_files[0].name
                        }
                    },
                    'num_images': 1,
                    'date_processed': datetime.now().isoformat()
                }
            }
        }
        
        # Use same dataset class for consistent preprocessing
        dataset = CellDataset(temp_data)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            image = dataset[0][0].unsqueeze(0).to(device)
            outputs = model(image)
            
            # Get probabilities and predictions
            probs = F.softmax(outputs, dim=1)
            pred_class = outputs.argmax(1).item()
            
            # Get class labels
            pred_label = dataset.label_encoder.inverse_transform([pred_class])[0]
            
            # Get probability distribution
            class_probs = {
                dataset.label_encoder.inverse_transform([i])[0]: f"{prob.item():.4f}"
                for i, prob in enumerate(probs[0])
            }
            
            return {
                'predicted_label': pred_label,
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


def calculate_class_weights(loader):
    """Calculate class weights for imbalanced dataset"""
    class_counts = defaultdict(int)
    for _, labels in loader:
        
        
        for label in labels:
            class_counts[int(label)] += 1
    
    total_samples = sum(class_counts.values())
    ordered_weights = sorted(class_counts.items(), key=lambda x: x[0])
    class_weights = torch.FloatTensor([total_samples / count for _, count in ordered_weights])
  
    return class_weights


def train_model(model, train_loader, test_loader, num_epochs=30, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Efficient optimizer setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,  # Higher initial learning rate
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Aggressive learning schedule for faster convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,  # Fast warmup
        div_factor=10,
        final_div_factor=100
    )
    
    scaler = GradScaler()
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    best_test_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = _train_epoch(
            model, train_loader, optimizer, criterion, 
            scaler, scheduler, device
        )
        
        model.eval()
        test_loss, test_acc = _evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logging.info(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
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
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logging.info(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        if (epoch + 1) % 2 == 0:
            plot_training_curves(
                train_losses, train_accuracies,
                test_losses, test_accuracies,
                Path('outputs'), epoch
            )
    
    return model


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
        
        current_lr = scheduler.get_last_lr()[0]
        progress_bar.set_postfix({
            'loss': f'{running_loss/(progress_bar.n+1):.3f}',
            'acc': f'{100.*correct/total:.1f}%',
            'lr': f'{current_lr:.2e}'
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
            batch_size = 32,
            shuffle = True,
            num_workers = 4,
            pin_memory = True,
            persistent_workers = True,
            prefetch_factor = 2,
            drop_last = True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size = 32,
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
            num_epochs = 30,
            device = device
        )
        
        results = model_inference(
            trained_model, 
            'ENSG00000184743', 
            device = device
        )
        logging.info(f"Inference results: {results}")
        
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
        raise
