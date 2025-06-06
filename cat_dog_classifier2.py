import os
import numpy as np
import warnings
from PIL import Image, ImageFile
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.metrics import recall_score
from tqdm import tqdm

from utils import PET_DATA_DIR, get_device

# Suppress PIL warnings and enable truncated image loading
warnings.filterwarnings(
    "ignore", message="Truncated File Read", module="PIL.TiffImagePlugin"
)
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True

# Hyperparameters
# Dataset directory can be configured via the ``PET_DATA_DIR`` environment variable.
DATA_DIR = PET_DATA_DIR
BATCH_SIZE = 32
NUM_WORKERS = 8
LR = 0.004
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
PATIENCE = 10
GRAD_CLIP = 5.0
WARMUP_EPOCHS = 4

# For the lightweight model, we use a single dropout rate:
DROP_RATE = 0.3


# Data-Related Functions
class ConvertToRGB:
    def __call__(self, image):
        return image.convert("RGB")


def create_train_transforms():
    return transforms.Compose(
        [
            transforms.Resize((170, 170)),
            transforms.RandomCrop((160, 160)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            ConvertToRGB(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def create_test_transforms():
    return transforms.Compose(
        [
            transforms.Resize((128, 128)),
            ConvertToRGB(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def create_datasets_from_folder(data_dir, train_transform, test_transform):
    base_dataset = datasets.ImageFolder(root=data_dir, transform=None)
    num_classes = len(base_dataset.classes)
    print("Classes:", base_dataset.classes)
    print("Class to idx mapping:", base_dataset.class_to_idx)
    print("Total images before split:", len(base_dataset))

    indices = np.random.permutation(len(base_dataset))
    train_size = int(0.8 * len(base_dataset))
    val_size = int(0.1 * len(base_dataset))
    test_size = len(base_dataset) - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[-test_size:]

    train_dataset = Subset(
        datasets.ImageFolder(data_dir, transform=train_transform), train_indices
    )
    val_dataset = Subset(
        datasets.ImageFolder(data_dir, transform=test_transform), val_indices
    )
    test_dataset = Subset(
        datasets.ImageFolder(data_dir, transform=test_transform), test_indices
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))
    return train_dataset, val_dataset, test_dataset, num_classes


def create_dataloaders(
    train_dataset, val_dataset, test_dataset, batch_size, num_workers
):
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
        "persistent_workers": True,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


# Lightweight CNN Model Definition
class LightweightCNN(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.3):
        super().__init__()
        self.activation = nn.ReLU()

        # Block 1: 3 -> 64 filters (increased from 32)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=drop_rate),
        )
        # Block 2: 64 -> 128 filters (increased from 64)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=drop_rate),
        )
        # Block 3: 128 -> 256 filters (increased from 128)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=drop_rate),
        )
        # Global Average Pooling and final classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(256, num_classes),  # Updated input features to 256
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Utility Functions
def compute_average_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    return total_loss / total_samples if total_samples > 0 else float("inf")


def compute_accuracy_and_recall(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            with autocast():
                outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    accuracy = correct / total if total > 0 else 0.0
    recall = recall_score(all_labels, all_preds, average="macro")
    return accuracy, recall


def create_model(num_classes, device, drop_rate=DROP_RATE):
    model = LightweightCNN(num_classes=num_classes, drop_rate=drop_rate).to(device)
    return model


def create_optimizer(model, lr, weight_decay):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_criterion():
    return nn.CrossEntropyLoss()


# Warmup Scheduler
def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr):
    # Linearly increase LR from 0 to base_lr during warmup_epochs
    lr = base_lr * (epoch + 1) / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# Training Function with Mixed Precision, Warmup, and LR Scheduling
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience,
    grad_clip,
    warmup_epochs,
):
    best_val_loss = float("inf")
    no_improve = 0
    scaler = GradScaler()  # Initialize gradient scaler

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        )

        # Apply warmup for the first warmup_epochs
        if epoch < warmup_epochs:
            adjust_learning_rate(optimizer, epoch, warmup_epochs, LR)

        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / total if total > 0 else float("inf")
        train_acc = correct / total if total > 0 else 0.0
        val_loss = compute_average_loss(model, val_loader, criterion, device)

        # Step the scheduler after warmup is finished
        if epoch >= warmup_epochs:
            scheduler.step()

        # Optionally print the current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


# Main Function
def main():
    device = get_device()
    print(f"Number of CPU cores available: {os.cpu_count()}")

    train_transform = create_train_transforms()
    test_transform = create_test_transforms()

    train_dataset, val_dataset, test_dataset, num_classes = create_datasets_from_folder(
        DATA_DIR, train_transform, test_transform
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, BATCH_SIZE, NUM_WORKERS
    )

    sample_img, sample_label = next(iter(train_loader))
    print(
        f"Sample batch images shape: {sample_img.shape}, Sample batch labels shape: {sample_label.shape}"
    )

    # Create the model using LightweightCNN
    model = create_model(num_classes=num_classes, device=device)

    optimizer = create_optimizer(model, LR, WEIGHT_DECAY)
    # Initialize cosine annealing scheduler (for epochs after warmup)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS
    )
    criterion = create_criterion()

    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        NUM_EPOCHS,
        PATIENCE,
        GRAD_CLIP,
        WARMUP_EPOCHS,
    )

    test_acc, test_recall = compute_accuracy_and_recall(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}, Test Recall: {test_recall:.4f}")


if __name__ == "__main__":
    main()
