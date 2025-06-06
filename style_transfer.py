import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt

# ------------------------------
# Low-level utility functions
# ------------------------------

def load_image(image_path, max_size=256, device=torch.device("cpu")):
    """
    Load an image, apply EXIF orientation, resize while preserving aspect ratio, 
    and preprocess it for VGG19.
    """
    image = Image.open(image_path).convert('RGB')
    # Apply the correct orientation using EXIF data, if available.
    image = ImageOps.exif_transpose(image)
    
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # add batch dimension
    return image.to(device)


def save_image(tensor, path):
    """
    Save the stylized image tensor to a file.

    Args:
        tensor (torch.Tensor): Image tensor to save.
        path (str): Output file path.
    """
    image = tensor.clone().detach().cpu().squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


def gram_matrix(feature_map):
    """
    Compute the Gram matrix for a feature map.

    Args:
        feature_map (torch.Tensor): Feature map tensor.

    Returns:
        torch.Tensor: Normalized Gram matrix.
    """
    b, c, h, w = feature_map.size()
    features = feature_map.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram.div(b * c * h * w)


# ------------------------------
# Low-level classes
# ------------------------------

class VGGFeatures(nn.Module):
    """
    Extract features from specified VGG19 layers.

    Args:
        layers (list): List of layer indices (as strings) to extract features from.
        device (torch.device): Device on which to load the VGG model.
    """
    def __init__(self, layers, device=torch.device("cpu")):
        super(VGGFeatures, self).__init__()
        # Load the pretrained VGG19 model with ImageNet weights
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.layers = layers
        self.model = nn.Sequential()
        max_layer = max(int(layer) for layer in layers)
        for i, layer in enumerate(vgg):
            if i > max_layer:
                break
            self.model.add_module(str(i), layer)

    def forward(self, x):
        """
        Forward pass to extract features from specified layers.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            dict: Dictionary mapping layer indices to features.
        """
        outputs = {}
        for name, layer in self.model.named_children():
            x = layer(x)
            if name in self.layers:
                outputs[name] = x
        return outputs


class ContentLoss(nn.Module):
    """
    Compute content loss between input and target features.

    Args:
        target (torch.Tensor): Target content features.
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        return nn.functional.mse_loss(input, self.target)


class StyleLoss(nn.Module):
    """
    Compute style loss between input Gram matrix and target Gram matrix.

    Args:
        target_gram (torch.Tensor): Target Gram matrix.
    """
    def __init__(self, target_gram):
        super(StyleLoss, self).__init__()
        self.target = target_gram.detach()

    def forward(self, input):
        gram = gram_matrix(input)
        return nn.functional.mse_loss(gram, self.target)


# ------------------------------
# High-level main execution logic
# ------------------------------

def main():
    # Hyperparameters
    content_weight = 0.8
    style_weight = 18e6
    num_steps = 100
    content_max_size = 512
    style_max_size = 128
    content_layers = ['21']           # corresponds to 'conv4_2' in VGG19
    style_layers = ['1', '6', '11', '20', '29']  # e.g., 'conv1_1', 'conv2_1', etc.

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define image paths
    content_path = r"C:\Users\jimen\OneDrive\Desktop\neuralnetworks\cat2.jpeg"
    style_folder = r"C:\Users\jimen\OneDrive\Desktop\neuralnetworks\picasso"

    # Load content image
    content_img = load_image(content_path, max_size=content_max_size, device=device)
    print(f"Loaded content image from {content_path}")

    # Load style image paths
    style_paths = [os.path.join(style_folder, f) for f in os.listdir(style_folder)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(style_paths)} Picasso style images.")

    # Create VGG feature extractor using both content and style layers
    vgg = VGGFeatures(style_layers + content_layers, device=device)

    # Compute content features
    content_features = vgg(content_img)

    # Accumulate Gram matrices for style images
    accum_grams = {layer: None for layer in style_layers}
    num_styles = len(style_paths)

    for path in style_paths:
        style_img = load_image(path, max_size=style_max_size, device=device)
        features = vgg(style_img)
        for layer in style_layers:
            gram = gram_matrix(features[layer])
            if accum_grams[layer] is None:
                accum_grams[layer] = gram
            else:
                accum_grams[layer] += gram
        del style_img, features
        torch.cuda.empty_cache()

    # Average the Gram matrices across all style images
    style_grams = {layer: accum_grams[layer] / num_styles for layer in style_layers}

    # Initialize loss modules
    content_loss = ContentLoss(content_features[content_layers[0]])
    style_losses = {layer: StyleLoss(style_grams[layer]) for layer in style_layers}

    # Initialize target image (starting from the content image)
    target_img = content_img.clone().requires_grad_(True)

    # Set up the optimizer
    optimizer = optim.LBFGS([target_img])

    # Optimization loop
    print("Starting style transfer optimization...")
    for step in range(num_steps):
        def closure():
            optimizer.zero_grad()
            features = vgg(target_img)
            c_loss = content_loss(features[content_layers[0]])
            s_loss = sum(style_losses[layer](features[layer]) for layer in style_layers) / len(style_layers)
            total_loss = content_weight * c_loss + style_weight * s_loss
            total_loss.backward(retain_graph=True)
            return total_loss

        optimizer.step(closure)

        if step % 10 == 0:
            current_loss = closure().item()
            print(f"Step {step}/{num_steps}, Loss: {current_loss}")
        torch.cuda.empty_cache()

    # Save and display the final stylized image
    output_path = r"C:\Users\jimen\OneDrive\Desktop\neuralnetworks\dog_picasso_style.jpg"
    save_image(target_img, output_path)
    print(f"Stylized image saved as {output_path}")

    plt.imshow(Image.open(output_path))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
