import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from utils import get_device

# Device configuration
device = get_device()


def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float() * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


# Load MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Hyperparameters
T = 1000
beta_start = 0.0001
beta_end = 0.1
img_size = 28
channels = 1

betas = torch.linspace(beta_start, beta_end, T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


def forward_diffusion(x0, t, noise):
    sqrt_alpha_prod = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    xt = sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise
    return xt


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
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        t_emb_proj = self.time_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb_proj
        h = self.conv2(h)
        h = self.bn2(h)
        return self.relu(h + self.res_conv(x))


class ImprovedUNet(nn.Module):
    def __init__(self, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.down1 = ResidualBlock(1, 64, time_emb_dim)
        self.down2 = ResidualBlock(64, 128, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(128, 256, time_emb_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up1 = ResidualBlock(256 + 128, 128, time_emb_dim)
        self.up2 = ResidualBlock(128 + 64, 64, time_emb_dim)
        self.conv_final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        b = self.bottleneck(self.pool(d2), t_emb)
        up1 = self.upsample(b)
        up1 = torch.cat([up1, d2], dim=1)
        up1 = self.up1(up1, t_emb)
        up2 = self.upsample(up1)
        up2 = torch.cat([up2, d1], dim=1)
        up2 = self.up2(up2, t_emb)
        out = self.conv_final(up2)
        return out


model = ImprovedUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


def train_model(epochs=50):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (x0, _) in enumerate(train_loader):
            x0 = x0.to(device)
            batch_size = x0.shape[0]
            t = torch.randint(0, T, (batch_size,), device=device).long()
            noise = torch.randn_like(x0).to(device)
            xt = forward_diffusion(x0, t, noise)
            predicted_noise = model(xt, t)
            loss = nn.MSELoss()(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")


train_model()


def sample(n_samples=16):
    model.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, channels, img_size, img_size).to(device)
        for t in reversed(range(T)):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_tensor)
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x = (1 / torch.sqrt(alpha_t)) * (
                x - ((beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)
            ) + torch.sqrt(beta_t) * noise
        x = (x * 0.5 + 0.5).clamp(0, 1)
        return x


samples = sample()
torchvision.utils.save_image(samples, "improved_samples.png", nrow=4)
