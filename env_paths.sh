#!/bin/bash
# Environment variables for neural network example scripts

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python script paths
export CAT_DOG_CLASSIFIER2_PY="$SCRIPT_DIR/cat_dog_classifier2.py"
export DIFFUSION_PY="$SCRIPT_DIR/diffusion.py"
export DIFFUSION2_PY="$SCRIPT_DIR/diffusion2.py"
export DIFFUSION3_PY="$SCRIPT_DIR/diffusion3.py"
export STYLE_TRANSFER_PY="$SCRIPT_DIR/style_transfer.py"
export STYLE_TRANSFER2_PY="$SCRIPT_DIR/style_transfer2.py"
export TRANSFORMER_PY="$SCRIPT_DIR/transformer.py"
export VAE_PY="$SCRIPT_DIR/vae.py"

# Usage message
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script sets environment variables for each example Python script."
    echo "Run 'source env_paths.sh' to populate the variables in your shell."
fi
