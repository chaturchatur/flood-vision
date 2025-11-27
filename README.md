# FloodVision: Multimodal Flood Detection

A deep learning pipeline for flood detection using **Sentinel-1 (SAR)** and **Sentinel-2 (Optical)** satellite imagery. This project implements a **Gated Fusion Network** to robustly detect water even in cloudy conditions by leveraging the all-weather capabilities of SAR.

## Features

- **Multimodal Data Loading**: Handles S1 (Radar), S2 (Optical), and Flood Masks.
- **Geographic Splitting**: Ensures zero overlap between Train/Test regions (e.g., Train on Americas, Test on Asia).
- **Robust Preprocessing**: Automatic normalization and dB scaling for SAR data.
- **Deep Learning**: Implementations of Encoder-Decoder architectures (UNet/ResNet).

## Project Structure

```text
flood_vision/
├── data/                   # Dataset storage (downloaded automatically)
│   └── sen1floods11/       # Sen1Floods11 Dataset
│       └── splits/         # CSV files defining Train/Test sets
├── src/
│   ├── data/
│   │   ├── download_datasets.py  # Script to fetch data
│   │   ├── dataset.py            # PyTorch Dataset class
│   │   ├── preprocessing.py      # Split generation script
│   │   └── transforms.py         # Albumentations
│   ├── models/                   # Neural Network architectures
│   └── training/                 # Training loops and metrics
├── notebooks/              # Jupyter notebooks for exploration
└── requirements.txt        # Python dependencies
```

## Setup & Installation

1.  **Clone the repo:**

    ```bash
    git clone https://github.com/yourusername/flood-vision.git
    cd flood-vision
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    _Note: You may need to install `gdal` or `rasterio` binaries separately depending on your OS._

3.  **Download Data:**

    ```bash
    # Download Sen1Floods11 (requires gsutil)
    python src/data/download_datasets.py sen1floods11
    ```

4.  **Generate Splits:**
    ```bash
    # Creates train_split.csv and test_split.csv
    python src/data/preprocessing.py
    ```

## Dataset

The project primarily uses **Sen1Floods11**, a benchmark dataset containing co-registered Sentinel-1 and Sentinel-2 imagery with hand-labeled water masks.

- **Train Set**: ~280 images (Bolivia, Ghana, Nigeria, Paraguay, Somalia, Spain, USA)
- **Test Set**: ~170 images (India, Cambodia, Pakistan, Sri Lanka, Mekong)

## Model Architecture (Planned)

- **Encoders**: Dual ResNet34 backbones (one for S1, one for S2).
- **Fusion**: Gated Fusion Module to suppress Optical features when clouds are detected.
- **Decoder**: UNet decoder outputting a binary flood mask.
