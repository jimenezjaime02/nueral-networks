import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image
import os
from tqdm import tqdm  # For the progress bar

from utils import PET_DATA_DIR, get_device


# ======================
# Configuration
# ======================
class Config:
    # Data parameters
    IMG_SIZE = 32
    CHANNELS = 3
    # Dataset directory configurable via the ``PET_DATA_DIR`` environment variable.
    DATA_PATH = PET_DATA_DIR
    BATCH_SIZE = 64
    NUM_WORKERS = 8

    # Model parameters
    TIME_EMB_DIM = 256
    BASE_CHANNELS = 256

    # Training parameters
    EPOCHS = 50
    LR = 0.0001

    # Diffusion process hyperparameters
    T = 2000
    BETA_START = 0.0001
    BETA_END = 0.02

    # Sampling parameters
    N_SAMPLES = 16


# ======================
# Diffusion Utilities
# ======================
def get_timestep_embedding(timesteps, embedding_dim):
    """
    Returns a sinusoidal embedding of the timesteps, similar to positional encodings in Transformers.
    """
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float() * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def compute_diffusion_params(config, device):
    """
    Compute beta, alpha, and related precomputed terms for the forward diffusion process.
    """
    T = config.T
    betas = torch.linspace(config.BETA_START, config.BETA_END, T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return (
        betas,
        alphas,
        alphas_cumprod,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
    )


def forward_diffusion(x0, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    Forward diffusion: x_t = sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise.
    """
    sqrt_alpha_prod = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    xt = sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise
    return xt


# ======================
# Self-Attention Module
# ======================
class SelfAttention(nn.Module):
    """
    A multi-head self-attention module for 2D feature maps.
    """

    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads

        # Each head gets head_dim = in_channels // num_heads
        assert (
            in_channels % num_heads == 0
        ), "in_channels must be divisible by num_heads"
        self.head_dim = in_channels // num_heads

        # Convolutional projections for Q, K, V (1x1 conv = linear layer in 2D)
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Output projection after combining attention heads
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 1) Project Q, K, V
        q = self.query(x)  # (B, C, H, W)
        k = self.key(x)
        v = self.value(x)

        # 2) Reshape to split heads: (B, num_heads, head_dim, H*W)
        q = q.reshape(B, self.num_heads, self.head_dim, -1)
        k = k.reshape(B, self.num_heads, self.head_dim, -1)
        v = v.reshape(B, self.num_heads, self.head_dim, -1)

        # 3) For matmul, rearrange q -> (B, num_heads, H*W, head_dim), k -> (B, num_heads, head_dim, H*W)
        q = q.permute(0, 1, 3, 2)  # (B, num_heads, H*W, head_dim)
        k = k.permute(0, 1, 2, 3)  # (B, num_heads, head_dim, H*W)

        # 4) Compute attention scores
        attn_scores = torch.matmul(q, k) / (self.head_dim**0.5)
        attn_prob = torch.softmax(attn_scores, dim=-1)  # (B, num_heads, H*W, H*W)

        # 5) Apply attention to values
        #    v -> (B, num_heads, H*W, head_dim)
        v = v.permute(0, 1, 3, 2)
        attn_out = torch.matmul(attn_prob, v)  # (B, num_heads, H*W, head_dim)

        # 6) Reshape back to (B, C, H, W)
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        # 7) Final projection + residual
        out = self.out_proj(attn_out)
        return x + out


# ======================
# Model Definition
# ======================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, t_emb):
        """
        x: (B, in_channels, H, W)
        t_emb: (B, time_emb_dim)
        """
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)

        # Project time embedding to match out_channels, then broadcast
        t_emb_proj = self.time_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb_proj

        h = self.conv2(h)
        h = self.bn2(h)

        return self.relu(h + self.res_conv(x))


class ImprovedUNet(nn.Module):
    """
    UNet with Residual Blocks, Time Embedding, and an Attention Block in the bottleneck.
    """

    def __init__(self, time_emb_dim=128, channels=3, base_channels=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        # Time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Downsampling path
        self.down1 = ResidualBlock(channels, base_channels, time_emb_dim)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck: Residual block + Self-Attention
        self.bottleneck = ResidualBlock(
            base_channels * 2, base_channels * 4, time_emb_dim
        )
        self.self_attn = SelfAttention(base_channels * 4, num_heads=4)

        # Upsampling path
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up1 = ResidualBlock(
            base_channels * 4 + base_channels * 2, base_channels * 2, time_emb_dim
        )
        self.up2 = ResidualBlock(
            base_channels * 2 + base_channels, base_channels, time_emb_dim
        )
        self.conv_final = nn.Conv2d(base_channels, channels, kernel_size=1)

    def forward(self, x, t):
        # 1) Time Embedding
        t = t.unsqueeze(-1)  # shape: (B, 1)
        t_emb = get_timestep_embedding(t, self.time_emb_dim)  # shape: (B, time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        # 2) Downsample
        d1 = self.down1(x, t_emb)  # (B, base_channels,    H,   W)
        d2 = self.down2(self.pool(d1), t_emb)  # (B, base_channels*2,  H/2, W/2)

        # 3) Bottleneck
        b = self.bottleneck(self.pool(d2), t_emb)  # (B, base_channels*4, H/4, W/4)
        b = self.self_attn(b)  # Apply self-attention

        # 4) Upsample
        up1 = self.upsample(b)  # (B, base_channels*4, H/2, W/2)
        up1 = torch.cat([up1, d2], dim=1)  # Skip connection
        up1 = self.up1(up1, t_emb)  # (B, base_channels*2, H/2, W/2)

        up2 = self.upsample(up1)  # (B, base_channels*2, H,   W)
        up2 = torch.cat([up2, d1], dim=1)  # Skip connection
        up2 = self.up2(up2, t_emb)  # (B, base_channels,   H,   W)

        out = self.conv_final(up2)  # (B, channels, H, W)
        return out


# ======================
# Data Handling Utilities
# ======================
def safe_loader(path):
    """
    Safely load an image from disk. If there's an error, return None.
    """
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except Exception:
        return None


def get_transforms(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def load_dataset(data_path, transform):
    print(f"[INFO] Loading dataset from: {data_path}")
    dataset = torchvision.datasets.ImageFolder(
        root=data_path, transform=transform, loader=safe_loader
    )
    print(f"[INFO] Found classes: {dataset.classes}")
    print(f"[INFO] Total images in dataset: {len(dataset)}")
    return dataset


def filter_dog_dataset(dataset):
    """
    Keep only 'Dog' images from the dataset.
    """
    if "Dog" not in dataset.class_to_idx:
        raise ValueError("[ERROR] 'Dog' class not found in dataset.")
    dog_class_index = dataset.class_to_idx["Dog"]
    dog_indices = []
    for i, (img, label) in enumerate(dataset):
        if img is not None and label == dog_class_index:
            dog_indices.append(i)
    dog_dataset = Subset(dataset, dog_indices)
    print(f"[INFO] Total dog images: {len(dog_dataset)}")
    return dog_dataset


def create_dataloader(dataset, batch_size, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


# ======================
# Training and Sampling
# ======================
def train_model(
    model,
    train_loader,
    optimizer,
    config,
    device,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
):
    """
    Main training loop for the diffusion model.
    Each iteration:
      - Sample a random timestep t
      - Add noise to the real image according to q(x_t | x_0)
      - The model predicts the noise
      - Compute MSE loss and update
    """
    model.train()
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print("[INFO] Starting training...")

    for epoch in range(config.EPOCHS):
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{config.EPOCHS}",
            leave=True,
        )

        for batch_idx, (x0, _) in progress_bar:
            if x0 is None:
                continue

            x0 = x0.to(device, non_blocking=True)
            x0 = x0.contiguous(memory_format=torch.channels_last)

            batch_size = x0.shape[0]
            t = torch.randint(0, config.T, (batch_size,), device=device).long()
            noise = torch.randn_like(x0)

            # Forward diffusion to get x_t
            xt = forward_diffusion(
                x0, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
            )

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                predicted_noise = model(xt, t)
                loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    print("[INFO] Training complete.")


def sample(model, config, device, betas, alphas, alphas_cumprod):
    """
    Use the reverse (denoising) diffusion process to sample images from noise.
    """
    model.eval()
    print("[INFO] Starting sampling...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            x = torch.randn(
                config.N_SAMPLES,
                config.CHANNELS,
                config.IMG_SIZE,
                config.IMG_SIZE,
                device=device,
            )
            x = x.contiguous(memory_format=torch.channels_last)

            for t in reversed(range(config.T)):
                t_tensor = torch.full(
                    (config.N_SAMPLES,), t, device=device, dtype=torch.long
                )
                predicted_noise = model(x, t_tensor)
                beta_t = betas[t]
                alpha_t = alphas[t]
                alpha_cumprod_t = alphas_cumprod[t]

                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (1 / torch.sqrt(alpha_t)) * (
                    x - ((beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)
                ) + torch.sqrt(beta_t) * noise

        # Map from [-1, 1] to [0, 1]
        x = (x * 0.5 + 0.5).clamp(0, 1)
    print("[INFO] Sampling complete.")
    return x


# ======================
# Main Execution Pipeline
# ======================
def main():
    config = Config()
    device = get_device()

    # Compute diffusion parameters
    (
        betas,
        alphas,
        alphas_cumprod,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
    ) = compute_diffusion_params(config, device)

    # Create dataset and dataloader
    transform = get_transforms(config.IMG_SIZE)
    dataset = load_dataset(config.DATA_PATH, transform)
    dog_dataset = filter_dog_dataset(dataset)
    train_loader = create_dataloader(dog_dataset, config.BATCH_SIZE, config.NUM_WORKERS)

    # Initialize model with self-attention in the bottleneck
    model = ImprovedUNet(
        time_emb_dim=config.TIME_EMB_DIM,
        channels=config.CHANNELS,
        base_channels=config.BASE_CHANNELS,
    ).to(device)

    # Use channels_last for potential speedup
    model = model.to(memory_format=torch.channels_last)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # Train
    train_model(
        model,
        train_loader,
        optimizer,
        config,
        device,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
    )

    # Sample from the diffusion model
    samples = sample(model, config, device, betas, alphas, alphas_cumprod)
    torchvision.utils.save_image(samples, "improved_dog_samples.png", nrow=4)
    print("[INFO] Sampled images saved as 'improved_dog_samples.png'.")


if __name__ == "__main__":
    main()
