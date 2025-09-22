# Gradio Photomosaic App

Interactive photomosaic generator with performance analysis.
## Overview

Creates artistic mosaics by replacing image regions with matching tile images. Automatically generates 16×16, 32×32, and 64×64 grid mosaics with comprehensive performance analysis.

## Files

- `quantized_tiles_generation.py` - Downloads and processes tile library from Kaggle dataset
- `app.py` - Main mosaic generator with Gradio interface
- `requirements.txt` - Python dependencies

## Setup

```bash
# Clone and setup
git clone https://github.com/shrutipanicker2810/gradio-photomosaic-app.git
cd gradio-photomosaic-app

# Create virtual environment
python -m venv py_env
py_env\Scripts\activate  # Windows
# source py_env/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Generate tile library (one-time setup)
python quantized_tiles_generation.py

# Run application
python app.py
```

Access at: `http://127.0.0.1:7860`

## Usage

1. Upload an image
2. Adjust quantization levels (4-16)
3. Click "Generate Mosaics"
4. View results: Original → Quantized → Three mosaic variations
5. Check terminal for performance analysis

## Performance Results

Based on 800×800 images with 14,389 tiles:

| Grid Size | Time   | Tiles/sec | MSE    | SSIM  |
|-----------|--------|-----------|--------|-------|
| 16×16     | 147.8s | 1.7       | 2749.3 | 0.486 |
| 32×32     | 49.9s  | 20.5      | 2294.0 | 0.553 |
| 64×64     | 47.0s  | 87.2      | 2294.2 | 0.534 |

## Key Findings

- **Super-linear scaling**: 32×32 grids are 12× more efficient than 16×16
- **Optimal quality**: 32×32 configuration provides best SSIM score
- **Vectorization benefits**: Orders of magnitude faster than loop-based implementations

## Dataset Attribution

Tile images generated from the **Abstract Paintings Dataset** by flash10042:
- **Kaggle Dataset**: https://www.kaggle.com/datasets/flash10042/abstract-paintings-dataset
- **License**: Creative Commons
- **Usage**: Tile images are quantized and processed for mosaic generation

## Requirements

- Python 3.7+
- PIL, NumPy, SciPy, Gradio
- 4GB+ RAM for large tile sets

## License

Educational use - CS5130 coursework.
