import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# ------------------------------
# Utility Functions
# ------------------------------

def load_image(image_path, max_size=256, device=torch.device("cpu")):
    """
    Load an image, apply EXIF orientation, resize while preserving aspect ratio, 
    and preprocess it for VGG19.
    """
    image = Image.open(image_path).convert('RGB')
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
    image = transform(image).unsqueeze(0)
    return image.to(device)

def save_image(tensor, path):
    """
    Save the stylized image tensor to a file.
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
    """
    b, c, h, w = feature_map.size()
    features = feature_map.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram.div(b * c * h * w)

# ------------------------------
# Classes
# ------------------------------

class VGGFeatures(nn.Module):
    """
    Extract features from specified VGG19 layers.
    """
    def __init__(self, layers, device=torch.device("cpu")):
        super(VGGFeatures, self).__init__()
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
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        return nn.functional.mse_loss(input, self.target)

class StyleLoss(nn.Module):
    """
    Compute style loss between input Gram matrix and target Gram matrix.
    """
    def __init__(self, target_gram):
        super(StyleLoss, self).__init__()
        self.target = target_gram.detach()

    def forward(self, input):
        gram = gram_matrix(input)
        return nn.functional.mse_loss(gram, self.target)

# ------------------------------
# Main Execution Logic
# ------------------------------

def main():
    # Hyperparameters
    content_weight = 0.5
    style_weight = 18e6  # Reduced style weight to prevent large gradients
    num_steps = 40000
    content_max_size = 512  # Increased for higher-resolution content
    style_max_size = 128     # Increased for higher-resolution style
    content_layers = ['21'] # conv4_2 in VGG19
    style_layers = ['1', '6', '11', '20', '29']  # Style layers

    # Early stopping parameters
    early_stop_patience = 100  # Number of steps to wait for significant improvement
    improvement_threshold = 1e-4  # Relative improvement threshold (0.01%)
    best_loss = float('inf')
    patience_counter = 0

    # Paths (update these to your local paths)
    content_path = r"C:\Users\jimen\OneDrive\Desktop\neuralnetworks\cat2.jpeg"
    style_folder = r"C:\Users\jimen\OneDrive\Desktop\neuralnetworks\picasso"
    output_path = r"C:\Users\jimen\OneDrive\Desktop\neuralnetworks\dog_picasso_style.jpg"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return
    print(f"Using device: {device}")

    # Load content image
    content_img = load_image(content_path, max_size=content_max_size, device=device)
    print(f"Loaded content image from {content_path}")

    # Create VGG feature extractor
    vgg = VGGFeatures(style_layers + content_layers, device=device)

    # Compute content features in full precision
    with torch.no_grad():
        content_features = vgg(content_img)

    # Load style image paths
    style_paths = [os.path.join(style_folder, f) for f in os.listdir(style_folder)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(style_paths)} style images.")

    # Accumulate Gram matrices for style images in batches
    accum_grams = {layer: None for layer in style_layers}
    num_styles = len(style_paths)
    batch_size = 50
    print(f"Processing {num_styles} style images in batches of {batch_size}...")
    for i in range(0, num_styles, batch_size):
        batch_paths = style_paths[i:i + batch_size]
        for path in batch_paths:
            style_img = load_image(path, max_size=style_max_size, device=device)
            with torch.no_grad():
                features = vgg(style_img)
                for layer in style_layers:
                    gram = gram_matrix(features[layer])
                    if accum_grams[layer] is None:
                        accum_grams[layer] = gram
                    else:
                        accum_grams[layer] += gram
            del style_img, features
            torch.cuda.empty_cache()
        print(f"Processed batch {i // batch_size + 1}/{(num_styles + batch_size - 1) // batch_size}")

    # Average the Gram matrices
    style_grams = {layer: accum_grams[layer] / num_styles for layer in style_layers}

    # Initialize loss modules
    content_loss = ContentLoss(content_features[content_layers[0]])
    style_losses = {layer: StyleLoss(style_grams[layer]) for layer in style_layers}

    # Initialize target image
    target_img = content_img.clone().requires_grad_(True)

    # Set up the optimizer (Adam with reduced learning rate)
    optimizer = optim.Adam([target_img], lr=0.001)

    # Optimization loop (running in full FP32 for stability)
    print("Starting style transfer optimization...")
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Forward pass in full precision
        features = vgg(target_img)
        c_loss = content_loss(features[content_layers[0]])
        s_loss = sum(style_losses[layer](features[layer]) for layer in style_layers) / len(style_layers)
        total_loss = content_weight * c_loss + style_weight * s_loss

        total_loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_([target_img], max_norm=1.0)

        optimizer.step()

        # Clamp the target image values to keep them in a stable range
        with torch.no_grad():
            target_img.clamp_(-1.5, 1.5)

        # Early stopping logic: Check for relative improvement
        current_loss = total_loss.item()
        if best_loss > 0 and (best_loss - current_loss) / best_loss < improvement_threshold:
            patience_counter += 1
        else:
            best_loss = current_loss
            patience_counter = 0

        if step % 10 == 0:
            print(f"Step {step}/{num_steps}, Loss: {current_loss:.4f}, Patience: {patience_counter}/{early_stop_patience}")
            if torch.isnan(total_loss):
                print("NaN detected in loss. Stopping optimization.")
                break

        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered at step {step} with loss {current_loss:.4f}")
            break

        torch.cuda.empty_cache()

    # Save and display the final stylized image
    save_image(target_img, output_path)
    print(f"Stylized image saved as {output_path}")

    plt.imshow(Image.open(output_path))
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
