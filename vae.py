import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse

from utils import MNIST_DIR, get_device

# Set device for computation (GPU if available, else CPU)
device = get_device()


# Define the VAE class
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) implementation in PyTorch.

    Args:
        input_dim (int): Dimension of the input data (e.g., 784 for MNIST).
        hidden_dims (list): List of hidden layer dimensions.
        latent_dim (int): Dimension of the latent space.
        output_dist (str): Output distribution ('bernoulli' or 'gaussian').
    """

    def __init__(self, input_dim, hidden_dims, latent_dim, output_dist="bernoulli"):
        super(VAE, self).__init__()
        self.output_dist = output_dist
        self.latent_dim = latent_dim

        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Mean and log-variance layers for the latent distribution
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

        # Build decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.decoder_linear = nn.Linear(prev_dim, input_dim)

        # Apply sigmoid for Bernoulli output; no activation for Gaussian
        if output_dist == "bernoulli":
            self.decoder_activation = nn.Sigmoid()
        elif output_dist == "gaussian":
            self.decoder_activation = nn.Identity()
        else:
            raise ValueError("Invalid output_dist: choose 'bernoulli' or 'gaussian'")

    def encode(self, x):
        """
        Encode input to latent distribution parameters (mean and log-variance).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            mu (torch.Tensor): Mean of the latent distribution.
            log_var (torch.Tensor): Log-variance of the latent distribution.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + sigma * epsilon, where epsilon ~ N(0, I).

        Args:
            mu (torch.Tensor): Mean of the latent distribution.
            log_var (torch.Tensor): Log-variance of the latent distribution.

        Returns:
            z (torch.Tensor): Sampled latent variable.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode latent sample to reconstructed input.

        Args:
            z (torch.Tensor): Latent variable tensor of shape (batch_size, latent_dim).

        Returns:
            recon_x (torch.Tensor): Reconstructed input tensor.
        """
        h = self.decoder_linear(z)
        recon_x = self.decoder_activation(h)
        return recon_x

    def forward(self, x):
        """
        Forward pass: encode, sample, and decode.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            recon_x (torch.Tensor): Reconstructed input.
            mu (torch.Tensor): Mean of the latent distribution.
            log_var (torch.Tensor): Log-variance of the latent distribution.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var


def vae_loss(recon_x, x, mu, log_var, output_dist="bernoulli"):
    """
    Compute the VAE loss (negative ELBO).

    Args:
        recon_x (torch.Tensor): Reconstructed input.
        x (torch.Tensor): Original input.
        mu (torch.Tensor): Mean of the latent distribution.
        log_var (torch.Tensor): Log-variance of the latent distribution.
        output_dist (str): Output distribution ('bernoulli' or 'gaussian').

    Returns:
        loss (torch.Tensor): Total loss (reconstruction + KL divergence).
        recon_loss (torch.Tensor): Reconstruction loss component.
        kl_loss (torch.Tensor): KL divergence component.
    """
    # Reconstruction loss
    if output_dist == "bernoulli":
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    elif output_dist == "gaussian":
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    else:
        raise ValueError("Invalid output_dist")

    # KL divergence loss (closed-form for Gaussian prior and posterior)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Total loss (negative ELBO)
    loss = recon_loss + kl_loss
    return loss, recon_loss, kl_loss


def train_vae(model, train_loader, optimizer, num_epochs, output_dist):
    """
    Train the VAE model.

    Args:
        model (VAE): VAE model instance.
        train_loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer (e.g., Adam).
        num_epochs (int): Number of training epochs.
        output_dist (str): Output distribution ('bernoulli' or 'gaussian').
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, log_var = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, log_var, output_dist)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(train_loader.dataset)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
            f"Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}"
        )


def generate_samples(model, num_samples):
    """
    Generate new samples from the VAE.

    Args:
        model (VAE): VAE model instance.
        num_samples (int): Number of samples to generate.

    Returns:
        samples (torch.Tensor): Generated samples.
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decode(z)
    return samples


def main():
    # Parse command-line arguments for configurability
    parser = argparse.ArgumentParser(description="Train a VAE in PyTorch.")
    parser.add_argument(
        "--input_dim",
        type=int,
        default=784,
        help="Input dimension (e.g., 784 for MNIST)",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[400, 200],
        help="Hidden layer dimensions (e.g., 400 200)",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=20, help="Latent space dimension"
    )
    parser.add_argument(
        "--output_dist",
        type=str,
        default="bernoulli",
        help="Output distribution (bernoulli or gaussian)",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    args = parser.parse_args()

    # Load and prepare the MNIST dataset (example; replace with your dataset)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
    )
    train_dataset = datasets.MNIST(
        root=MNIST_DIR, train=True, transform=transform, download=True
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize the model, optimizer, and train
    model = VAE(args.input_dim, args.hidden_dims, args.latent_dim, args.output_dist).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_vae(model, train_loader, optimizer, args.num_epochs, args.output_dist)

    # Generate samples (example usage)
    samples = generate_samples(model, num_samples=10)
    print("Generated samples shape:", samples.shape)


if __name__ == "__main__":
    main()
