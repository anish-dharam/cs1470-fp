# CNN-Based Wheat Futures Price Forecasting

This project implements the research paper "Forecasting Agriculture Commodity Futures Prices with Convolutional Neural Networks with Application to Wheat Futures" using TensorFlow/Keras. The model predicts wheat futures price direction 20 trading days ahead by combining satellite imagery (Landsat) with weather and historical price data.

## Project Structure

```
cs1470-fp/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # CNN, dense, and combined models
│   ├── training/       # Training loop and metrics
│   └── utils/          # Helper functions
├── scripts/            # Main execution scripts
├── notebooks/          # Jupyter notebooks for exploration
└── requirements.txt    # Dependencies
```

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model with dummy data:

```bash
python scripts/train.py
```

### Prediction

Make predictions on new data:

```bash
python scripts/predict.py
```

## Data

Currently uses dummy data for development. Future integration with Google Earth Engine API for real Landsat satellite imagery.

## Model Architecture

- **CNN Backbone**: Processes satellite imagery (128x128x3 RGB)
- **Dense Network**: Handles weather and historical price features
- **Fusion Layer**: Combines both networks for binary classification

## Evaluation Metrics

- Classification Accuracy
- Profit and Loss (PNL)
- Confusion Matrix
