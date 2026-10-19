#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "This script expects x86-64 Linux, such as Ubuntu in WSL."
    exit 1
fi

if [[ ! -f environment-brain.yml ]]; then
    echo "Missing environment-brain.yml. Run from a downloaded project."
    exit 1
fi

# Install the small system tools needed for setup if missing.
if ! command -v curl >/dev/null ||
   ! command -v git >/dev/null ||
   ! command -v bzip2 >/dev/null; then
    sudo apt-get update
    sudo apt-get install -y curl git bzip2 ca-certificates
fi

# Find an existing Conda installation.
if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    conda_base="$HOME/miniconda3"
else
    conda_base="$HOME/miniconda3"

    if [[ -e "$conda_base" ]]; then
        echo "$conda_base exists but does not contain a working Conda."
        echo "Resolve that installation before continuing."
        exit 1
    fi

    installer="$(mktemp)"
    trap 'rm -f -- "$installer"' EXIT

    echo "Downloading Miniconda..."
    curl --fail --location --retry 3 \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        --output "$installer"

    echo "Follow the installer prompts and review its licence."
    bash "$installer" -p "$conda_base"

    rm -f -- "$installer"
    trap - EXIT
fi

# Make Conda available inside this script.
source "$conda_base/etc/profile.d/conda.sh"

echo "Downloading the fly submodule..."
git submodule update --init --recursive

if conda run -n brain-fly python --version >/dev/null 2>&1; then
    echo "Using the existing brain-fly environment."
else
    echo "Creating brain-fly. Downloads may take a while..."
    conda env create -n brain-fly -f environment-brain.yml
fi

echo "Checking PyTorch and GPU access..."
conda run --no-capture-output -n brain-fly python -c "
import torch

print('PyTorch:', torch.__version__)
print('GPU available:', torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit('GPU check failed. Check the Windows NVIDIA driver and WSL setup.')

print('GPU:', torch.cuda.get_device_name(0))
"

echo
echo "Setup complete."
echo "For this terminal, run:"
printf 'source "%s/etc/profile.d/conda.sh"\n' "$conda_base"
echo "conda activate brain-fly"