# Neural Network Training Documentation

## Overview
This document provides comprehensive documentation for the `neural_network_training.py` script, which implements a PyTorch-based neural network for credit card fraud detection using Optuna for hyperparameter optimization and comprehensive evaluation metrics.

## Purpose
The script trains a multi-layer perceptron (MLP) neural network to detect fraudulent credit card transactions. It uses Optuna for automated hyperparameter tuning and PyTorch for flexible neural network implementation with GPU acceleration support.

## Dependencies
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import optuna
```

## Data Processing Pipeline

### 1. Data Loading and Preprocessing
```python
df = pd.read_csv("creditcard.csv")
X = df.drop('Class', axis=1)  # Features
y = df['Class']               # Target variable (0: Normal, 1: Fraud)
```

### 2. Feature Scaling
- **StandardScaler**: All features are standardized for neural network training
- **Rationale**: Neural networks are sensitive to input scale variations

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 3. Train-Test Split
- **Test Size**: 20% of data reserved for testing
- **Random State**: 42 (for reproducibility)
- **Stratification**: Maintains class distribution

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
```

### 4. PyTorch Tensor Conversion
```python
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
```

### 5. DataLoader Setup
```python
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Neural Network Architecture

### Model Structure
The neural network is a multi-layer perceptron with the following components:

```python
class TrialMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),    # First hidden layer
            nn.ReLU(),                       # ReLU activation
            nn.Dropout(dropout1),            # Dropout regularization
            nn.Linear(hidden1, hidden2),     # Second hidden layer
            nn.ReLU(),                       # ReLU activation
            nn.Dropout(dropout2),            # Dropout regularization
            nn.Linear(hidden2, 1),           # Output layer
            nn.Sigmoid()                     # Sigmoid activation for binary classification
        )
```

### Architecture Components
- **Input Layer**: 30 features (credit card transaction features)
- **Hidden Layer 1**: Configurable size (32-128 neurons)
- **Hidden Layer 2**: Configurable size (16-64 neurons)
- **Output Layer**: 1 neuron (binary classification)
- **Activation Functions**: ReLU for hidden layers, Sigmoid for output
- **Regularization**: Dropout layers to prevent overfitting

## Hyperparameter Optimization with Optuna

### Optuna Objective Function
```python
def objective(trial):
    # Hyperparameter suggestions
    hidden1 = trial.suggest_int("hidden1", 32, 128)
    hidden2 = trial.suggest_int("hidden2", 16, 64)
    dropout1 = trial.suggest_float("dropout1", 0.1, 0.5)
    dropout2 = trial.suggest_float("dropout2", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
```

### Hyperparameter Ranges
- **hidden1**: 32-128 neurons (first hidden layer)
- **hidden2**: 16-64 neurons (second hidden layer)
- **dropout1**: 0.1-0.5 dropout rate (first dropout layer)
- **dropout2**: 0.1-0.5 dropout rate (second dropout layer)
- **learning_rate**: 1e-4 to 1e-2 (logarithmic scale)
- **batch_size**: 512, 1024, or 2048 samples per batch

### Optimization Process
- **Trials**: 30 optimization trials
- **Direction**: Maximize ROC-AUC score
- **Early Stopping**: Built into Optuna's optimization process

## Training Process

### 1. Hyperparameter Tuning Phase
```python
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
```

**Training Configuration for Tuning:**
- **Epochs**: 20 (reduced for faster hyperparameter search)
- **Loss Function**: Binary Cross-Entropy (BCELoss)
- **Optimizer**: Adam optimizer
- **Evaluation**: ROC-AUC on test set

### 2. Final Model Training
```python
epochs = 50  # Extended training with best hyperparameters
```

**Final Training Configuration:**
- **Epochs**: 50 (extended training for better convergence)
- **Best Hyperparameters**: Selected from Optuna optimization
- **Progress Monitoring**: Loss printed every 5 epochs

## Model Evaluation

### Prediction Generation
```python
model.eval()
with torch.no_grad():
    y_pred_proba = model(X_test_tensor.to(device)).cpu().numpy().flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)
```

### Evaluation Metrics
1. **Classification Report**: Precision, recall, F1-score, support
2. **Confusion Matrix**: Visual representation of predictions vs actual
3. **ROC Curve**: True Positive Rate vs False Positive Rate
4. **Precision-Recall Curve**: Precision vs Recall for imbalanced data

## Performance Visualizations

### 1. Confusion Matrix
- **Purpose**: Shows actual vs predicted classifications
- **Visualization**: Heatmap with annotations
- **Colors**: Blue gradient for readability

### 2. ROC Curve
- **X-axis**: False Positive Rate
- **Y-axis**: True Positive Rate
- **AUC**: Area Under the Curve
- **Baseline**: Random classifier diagonal line

### 3. Precision-Recall Curve
- **X-axis**: Recall
- **Y-axis**: Precision
- **PR-AUC**: Area Under Precision-Recall Curve
- **Importance**: Better metric for imbalanced datasets

## Output Files

All results are saved in the `results_best/neural_network/` directory:

1. **`classification_report.txt`**: Detailed classification metrics
2. **`confusion_matrix.png`**: Confusion matrix visualization
3. **`roc_curve.png`**: ROC curve plot
4. **`precision-recall_curve.png`**: Precision-recall curve plot

## Key Features

### Strengths
- **Automated hyperparameter tuning**: Optuna provides efficient optimization
- **GPU acceleration**: Automatic CUDA detection and usage
- **Flexible architecture**: Easy to modify network structure
- **Regularization**: Dropout layers prevent overfitting
- **Comprehensive evaluation**: Multiple metrics and visualizations
- **Reproducible results**: Fixed random seeds

### Advanced Capabilities
- **Dynamic hyperparameter search**: Optuna adapts search strategy
- **Memory efficient**: DataLoader with configurable batch sizes
- **Scalable**: Can handle larger datasets with appropriate batch sizes
- **Extensible**: Easy to add more layers or change architecture

## Hardware Requirements

### Minimum Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 8GB+ recommended for large datasets
- **Storage**: Sufficient space for model checkpoints

### GPU Acceleration (Optional)
- **CUDA**: NVIDIA GPU with CUDA support
- **Memory**: 4GB+ VRAM recommended
- **Automatic Detection**: Script automatically uses GPU if available

## Usage Instructions

1. **Prerequisites**: 
   - Install PyTorch: `pip install torch`
   - Install Optuna: `pip install optuna`
   - Ensure `creditcard.csv` is in the same directory

2. **Execution**: 
   ```bash
   python neural_network_training.py
   ```

3. **Results**: Check the `results_best/neural_network/` directory for outputs

## Performance Expectations

- **Training Time**: 
  - Hyperparameter tuning: 10-30 minutes (30 trials)
  - Final training: 5-15 minutes (50 epochs)
- **Memory Usage**: Moderate (depends on batch size and network size)
- **Accuracy**: High potential performance with proper hyperparameter tuning
- **Scalability**: Good for medium to large datasets

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Slow convergence**: Increase learning rate or adjust architecture
3. **Overfitting**: Increase dropout rates or reduce network size
4. **Underfitting**: Increase network size or reduce regularization

### Optimization Tips
1. **Batch size tuning**: Larger batches for stability, smaller for memory efficiency
2. **Learning rate**: Use learning rate scheduling for better convergence
3. **Architecture**: Experiment with different layer sizes and depths
4. **Regularization**: Balance dropout rates for optimal performance

## Future Enhancements

1. **Advanced architectures**: 
   - Residual connections
   - Batch normalization
   - Different activation functions

2. **Training improvements**:
   - Learning rate scheduling
   - Early stopping
   - Model checkpointing

3. **Feature engineering**:
   - Automated feature selection
   - Feature interaction modeling

4. **Model interpretability**:
   - SHAP values
   - LIME explanations
   - Attention mechanisms

5. **Production deployment**:
   - Model serialization
   - API endpoints
   - Real-time inference
