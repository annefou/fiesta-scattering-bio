# FIESTA Scattering Bio — FOSCAT environment for steps 01 and 03.
#
# Step 02 (CNN inference) requires TensorFlow 2.19 + planktonclas which
# conflicts with this image's PyTorch stack. Run step 02 in the companion
# image ghcr.io/annefou/fiesta-decrop-reproduction and copy its output
# npz files into this repo's results/ directory before running step 03.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        p7zip-full \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "numpy>=1.26,<2.3" \
        scipy \
        scikit-learn \
        pandas \
        matplotlib \
        Pillow \
        torch \
        jupytext \
        nbclient \
        ipykernel \
        zenodo-get \
        "foscat @ git+https://github.com/annefou/FOSCAT.git@v0.1.0-cpu"

WORKDIR /app
COPY . /app

# Default: run step 01 (feature extraction). Override with
#   docker run ... python 03_stacking.py
CMD ["python", "01_scattering_features.py"]
