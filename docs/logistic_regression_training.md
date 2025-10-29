# Logistic Regression Training Documentation

## Overview
This document provides comprehensive documentation for the `logistic_regression_training.py` script, which implements a logistic regression model for credit card fraud detection using hyperparameter optimization and comprehensive evaluation metrics.

## Purpose
The script trains a logistic regression classifier to detect fraudulent credit card transactions using the credit card fraud dataset. It employs grid search cross-validation for hyperparameter optimization and generates detailed performance metrics and visualizations.

## Dependencies
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve,
    auc, roc_curve
)
```

## Data Processing Pipeline

### 1. Data Loading
```python
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)  # Features
y = df['Class']               # Target variable (0: Normal, 1: Fraud)
```

### 2. Feature Scaling
- **StandardScaler**: All features are standardized to have zero mean and unit variance
- **Rationale**: Logistic regression is sensitive to feature scales, especially when using regularization

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 3. Train-Test Split
- **Test Size**: 20% of data reserved for testing
- **Random State**: 42 (for reproducibility)
- **Stratification**: Maintains class distribution in both train and test sets

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
```

## Hyperparameter Optimization

### Grid Search Configuration
The script uses `GridSearchCV` with the following parameter grid:

```python
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],           # Regularization strength
    'solver': ['lbfgs', 'liblinear', 'saga'], # Optimization algorithms
    'penalty': ['l2'],                        # Regularization type
    'class_weight': ['balanced', None],       # Class balancing
    'max_iter': [100, 500, 1000]             # Maximum iterations
}
```

### Parameter Explanations
- **C**: Inverse regularization strength (smaller values = stronger regularization)
- **solver**: 
  - `lbfgs`: Good for small datasets, supports L2 penalty
  - `liblinear`: Fast for small datasets, supports both L1 and L2
  - `saga`: Supports both L1 and L2 penalties, good for large datasets
- **penalty**: L2 regularization (Ridge regression)
- **class_weight**: Handles class imbalance
  - `balanced`: Automatically adjusts weights inversely proportional to class frequencies
  - `None`: Equal weights for all classes
- **max_iter**: Maximum number of iterations for convergence

### Cross-Validation Setup
- **CV Folds**: 5-fold cross-validation
- **Scoring Metric**: ROC-AUC (Area Under ROC Curve)
- **Parallel Processing**: Uses all available CPU cores (`n_jobs=-1`)

## Model Training and Evaluation

### Best Model Selection
```python
grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=2
)
```

### Prediction Generation
```python
y_pred = best_model.predict(X_test)                    # Binary predictions
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probability scores
```

## Performance Metrics and Visualizations

### 1. Classification Report
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of actual occurrences of each class

### 2. Confusion Matrix
- **Purpose**: Shows actual vs predicted classifications
- **Visualization**: Heatmap with annotations
- **Colors**: Blue gradient for better readability

### 3. ROC Curve
- **X-axis**: False Positive Rate (1 - Specificity)
- **Y-axis**: True Positive Rate (Sensitivity)
- **AUC**: Area Under the Curve (higher is better)
- **Baseline**: Diagonal line representing random classifier

### 4. Precision-Recall Curve
- **X-axis**: Recall
- **Y-axis**: Precision
- **PR-AUC**: Area Under Precision-Recall Curve
- **Useful for**: Imbalanced datasets (like fraud detection)

## Output Files

All results are saved in the `results_best/logistic_regression/` directory:

1. **`best_hyperparameters.json`**: Best hyperparameters found by grid search
2. **`classification_report.txt`**: Detailed classification metrics
3. **`confusion_matrix.png`**: Confusion matrix visualization
4. **`roc_curve.png`**: ROC curve plot
5. **`precision_recall_curve.png`**: Precision-recall curve plot

## Key Features

### Strengths
- **Comprehensive hyperparameter tuning**: Tests multiple combinations systematically
- **Class imbalance handling**: Uses `class_weight='balanced'` option
- **Robust evaluation**: Multiple metrics and visualizations
- **Reproducible results**: Fixed random seeds
- **Efficient computation**: Parallel processing with all CPU cores

### Considerations
- **Computational cost**: Grid search can be time-consuming with large parameter spaces
- **Feature scaling**: All features are scaled, which may not be necessary for all features
- **Regularization**: Only L2 penalty is tested (L1 could be added for feature selection)

## Usage Instructions

1. **Prerequisites**: Ensure `creditcard.csv` is in the same directory
2. **Dependencies**: Install required packages from `requirements.txt`
3. **Execution**: Run the script with `python logistic_regression_training.py`
4. **Results**: Check the `results_best/logistic_regression/` directory for outputs

## Performance Expectations

- **Training Time**: Moderate (depends on parameter grid size and data size)
- **Memory Usage**: Low to moderate
- **Accuracy**: Good baseline performance for fraud detection
- **Interpretability**: High (linear model coefficients are interpretable)

## Troubleshooting

### Common Issues
1. **Convergence warnings**: Increase `max_iter` parameter
2. **Memory issues**: Reduce parameter grid size or use fewer CV folds
3. **Poor performance**: Try different solvers or regularization strengths

### Optimization Tips
1. **Feature engineering**: Consider creating new features
2. **Sampling techniques**: Try SMOTE or other oversampling methods
3. **Ensemble methods**: Combine with other algorithms
4. **Threshold tuning**: Adjust decision threshold based on business requirements

## Future Enhancements

1. **Feature selection**: Add L1 regularization for automatic feature selection
2. **Advanced preprocessing**: Implement more sophisticated feature engineering
3. **Model interpretability**: Add SHAP or LIME explanations
4. **Real-time prediction**: Implement streaming prediction pipeline
5. **A/B testing**: Framework for comparing different models in production
