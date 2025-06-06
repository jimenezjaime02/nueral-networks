import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image
from tqdm import tqdm

from utils import PET_DATA_DIR, get_device

# Force default dtype to float64
torch.set_default_dtype(torch.float64)


# ======================
# Configuration
# ======================
class Config:
    IMG_SIZE = 64
    CHANNELS = 3
    # Dataset directory configurable via the ``PET_DATA_DIR`` environment variable.
    DATA_PATH = PET_DATA_DIR
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    TIME_EMB_DIM = 128
    BASE_CHANNELS = 128
    EPOCHS = 10
    LR = 0.0001
    T = 1000
    N_SAMPLES = 16


# ======================
# Device Setup handled via utils.get_device


# ======================
# Diffusion Utilities
# ======================
def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb_factor = torch.log(torch.tensor(10000.0, device=timesteps.device))
    emb_factor = emb_factor / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb_factor)
    timesteps = timesteps.unsqueeze(1)
    emb = timesteps * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, device="cpu") / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)
    return betas


def compute_diffusion_params(config, device):
    T = config.T
    betas = cosine_beta_schedule(T).to(device)
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
    sqrt_alpha_prod = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    xt = sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise
    return xt


# ======================
# Residual Block
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
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.relu(self.bn1(self.conv1(x)))
        t_emb_proj = self.time_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb_proj
        h = self.bn2(self.conv2(h))
        return self.relu(h + self.res_conv(x))


# ======================
# Self-Attention Block
# ======================
class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        assert (
            in_channels % num_heads == 0
        ), "in_channels must be divisible by num_heads"
        self.head_dim = in_channels // num_heads

        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).reshape(B, self.num_heads, self.head_dim, -1)
        k = self.key(x).reshape(B, self.num_heads, self.head_dim, -1)
        v = self.value(x).reshape(B, self.num_heads, self.head_dim, -1)

        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 2, 3)
        attn_scores = torch.matmul(q, k) / (self.head_dim**0.5)
        attn_prob = torch.softmax(attn_scores, dim=-1)

        v = v.permute(0, 1, 3, 2)
        attn_out = torch.matmul(attn_prob, v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.out_proj(attn_out)
        return x + out


# ======================
# Improved UNet
# ======================
class ImprovedUNet(nn.Module):
    def __init__(self, time_emb_dim=256, channels=3, base_channels=256):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.down1 = ResidualBlock(channels, base_channels, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.self_attn_down = SelfAttention(base_channels * 2)

        self.bottleneck = ResidualBlock(
            base_channels * 2, base_channels * 4, time_emb_dim
        )
        self.self_attn_bottle = SelfAttention(base_channels * 4)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up1 = ResidualBlock(base_channels * 6, base_channels * 2, time_emb_dim)
        self.self_attn_up1 = SelfAttention(base_channels * 2)
        self.up2 = ResidualBlock(base_channels * 3, base_channels, time_emb_dim)

        self.conv_final = nn.Conv2d(base_channels, channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(get_timestep_embedding(t, self.time_emb_dim))
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        d2 = self.self_attn_down(d2)
        b = self.pool(d2)
        b = self.bottleneck(b, t_emb)
        b = self.self_attn_bottle(b)
        up1 = self.upsample(b)
        up1 = torch.cat([up1, d2], dim=1)
        up1 = self.up1(up1, t_emb)
        up1 = self.self_attn_up1(up1)
        up2 = self.upsample(up1)
        up2 = torch.cat([up2, d1], dim=1)
        up2 = self.up2(up2, t_emb)
        out = self.conv_final(up2)
        return out


# ======================
# Data Handling
# ======================
def get_transforms(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def load_dataset(data_path, transform):
    return torchvision.datasets.ImageFolder(root=data_path, transform=transform)


def filter_dog_dataset(dataset):
    dog_idx = dataset.class_to_idx.get("Dog", None)
    if dog_idx is None:
        return dataset
    indices = [i for i, (_, label) in enumerate(dataset) if label == dog_idx]
    dog_dataset = Subset(dataset, indices)
    return dog_dataset


def create_dataloader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


# ======================
# Training Function
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
    model.train()
    mse = nn.MSELoss()

    for epoch in range(config.EPOCHS):
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}"):
            images = images.to(device)
            batch_size = images.size(0)
            t = torch.randint(0, config.T, (batch_size,), device=device).long()
            noise = torch.randn_like(images)
            xt = forward_diffusion(
                images, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
            )
            pred_noise = model(xt, t)
            loss = mse(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {loss.item():.4f}")


# ======================
# Sampling Function
# ======================
@torch.no_grad()
def sample(model, config, device, betas, alphas, alphas_cumprod):
    model.eval()
    num_samples = config.N_SAMPLES
    x = torch.randn(
        num_samples, config.CHANNELS, config.IMG_SIZE, config.IMG_SIZE, device=device
    )

    for i in reversed(range(config.T)):
        t = torch.tensor([i] * num_samples, device=device).long()
        pred_noise = model(x, t)
        alpha = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]
        beta = betas[i]
        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        x = (1 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise
        ) + torch.sqrt(beta) * noise

    return x


# ======================
# Main
# ======================
def main():
    config = Config()
    device = get_device()
    (
        betas,
        alphas,
        alphas_cumprod,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
    ) = compute_diffusion_params(config, device)
    transform = get_transforms(config.IMG_SIZE)
    dataset = load_dataset(config.DATA_PATH, transform)
    dog_dataset = filter_dog_dataset(dataset)
    train_loader = create_dataloader(dog_dataset, config.BATCH_SIZE, config.NUM_WORKERS)

    model = (
        ImprovedUNet(
            time_emb_dim=config.TIME_EMB_DIM,
            channels=config.CHANNELS,
            base_channels=config.BASE_CHANNELS,
        )
        .to(device)
        .to(torch.float64)
    )

    model = model.to(memory_format=torch.channels_last)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    train_model(
        model,
        train_loader,
        optimizer,
        config,
        device,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
    )

    samples = sample(model, config, device, betas, alphas, alphas_cumprod)
    torchvision.utils.save_image(samples, "improved_dog_samples.png", nrow=4)
    print("[INFO] Sampled images saved as 'improved_dog_samples.png'.")


if __name__ == "__main__":
    main()
