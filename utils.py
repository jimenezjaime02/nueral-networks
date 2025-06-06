import os
import torch


def get_device():
    """Return CUDA device if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# Dataset paths configured via environment variables for flexibility
PET_DATA_DIR = os.getenv("PET_DATA_DIR", "./data/PetImages")
CONTENT_IMAGE = os.getenv("CONTENT_IMAGE", "./data/content.jpg")
STYLE_FOLDER = os.getenv("STYLE_FOLDER", "./data/styles")
OUTPUT_IMAGE = os.getenv("OUTPUT_IMAGE", "./data/output.jpg")
MNIST_DIR = os.getenv("MNIST_DIR", "./data")
