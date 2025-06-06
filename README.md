# nueral-networks

This repository includes example neural network scripts.

Several scripts expect dataset or image paths. These can now be configured via environment variables:

- `PET_DATA_DIR` - directory containing the PetImages dataset.
- `CONTENT_IMAGE` - path to a content image used by style transfer examples.
- `STYLE_FOLDER` - folder containing style images.
- `OUTPUT_IMAGE` - output path for style transfer results.
- `MNIST_DIR` - directory to download the MNIST dataset for the VAE example.

## Git setup helper

Run `setup_git.sh` in this directory to configure Git with your username, email, and remote. You'll still need to provide your GitHub username and personal access token the first time you push.

