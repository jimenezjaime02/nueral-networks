"""CIFAR-10 image classification training script.

This module defines a simple convolutional neural network (CNN) for the
CIFAR-10 dataset together with helper utilities for loading the dataset,
training the network, and evaluating accuracy on the validation split.

Usage example:
    python cifar10_classifier.py --epochs 10 --batch-size 128
"""

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class TrainingConfig:
    """Hyper-parameters and training options."""

    data_dir: str = "./data"
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    validation_split: float = 0.1


class CIFAR10CNN(nn.Module):
    """A small convolutional network for CIFAR-10."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def create_dataloaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders for CIFAR-10."""

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    dataset = datasets.CIFAR10(root=config.data_dir, train=True, download=True, transform=transform)
    val_size = int(len(dataset) * config.validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: str) -> float:
    """Train the model for a single epoch and return the average loss."""

    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    *,
    num_classes: int = 10,
    return_confusion_matrix: bool = False,
) -> Tuple[float, Optional[torch.Tensor]]:
    """Evaluate the model accuracy on the provided dataloader.

    Args:
        model: The neural network to evaluate.
        dataloader: Iterable of batches used for evaluation.
        device: Device used to perform computation.
        num_classes: Number of classes for the confusion matrix.
        return_confusion_matrix: When ``True`` also compute and return a confusion matrix.

    Returns:
        Tuple containing the accuracy and the confusion matrix (or ``None`` when not
        requested).
    """

    model.eval()
    correct = 0
    total = 0
    confusion_matrix = (
        torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
        if return_confusion_matrix
        else None
    )

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if confusion_matrix is not None:
                for true_label, pred_label in zip(targets.view(-1), predicted.view(-1)):
                    confusion_matrix[true_label.long(), pred_label.long()] += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, confusion_matrix


def train(config: TrainingConfig) -> None:
    """Main training loop for CIFAR-10 classification."""

    device = config.device
    train_loader, val_loader = create_dataloaders(config)

    model = CIFAR10CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    for epoch in range(1, config.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        val_accuracy, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{config.epochs}: loss={loss:.4f}, val_acc={val_accuracy:.4%}")

    final_accuracy, confusion_matrix = evaluate(
        model, val_loader, device, return_confusion_matrix=True
    )
    print("Final validation accuracy:", f"{final_accuracy:.4%}")
    print("Confusion matrix (rows=true labels, columns=predicted labels):")
    print(confusion_matrix.cpu())


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train a simple CIFAR-10 classifier")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to download CIFAR-10")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader worker processes")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of training data to use as validation",
    )
    args = parser.parse_args()

    return TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        validation_split=args.validation_split,
    )


if __name__ == "__main__":
    train(parse_args())
