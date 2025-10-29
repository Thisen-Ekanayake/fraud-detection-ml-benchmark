# XGBoost Training Documentation

## Overview
This document provides comprehensive documentation for the `xgboost_training.py` script, which implements an XGBoost classifier for credit card fraud detection using Optuna for hyperparameter optimization and comprehensive evaluation with progress tracking.

## Purpose
The script trains an XGBoost (eXtreme Gradient Boosting) model to detect fraudulent credit card transactions. It employs Optuna for automated hyperparameter tuning and provides detailed performance metrics, feature importance analysis, and comprehensive visualizations with progress tracking.

## Dependencies
```python
import pandas as pd
import numpy as np
import os
import optuna
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve,
    auc, roc_curve
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
```

## Data Processing Pipeline

### 1. Data Loading and Initial Analysis
```python
df = pd.read_csv('creditcard.csv')
print("Dataset shape:", df.shape)
print(df['Class'].value_counts())
```

**Output Analysis**:
- **Dataset shape**: (284,807, 31) - 284,807 transactions with 31 features
- **Class distribution**: Highly imbalanced (fraud cases are rare)

### 2. Feature and Target Separation
```python
X = df.drop('Class', axis=1)  # Features (30 columns)
y = df['Class']               # Target variable (0: Normal, 1: Fraud)
```

### 3. Selective Feature Scaling
```python
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
```

**Rationale**:
- XGBoost is tree-based and doesn't require feature scaling
- Only `Time` and `Amount` are scaled due to different scales
- PCA features (V1-V28) are already normalized

### 4. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- **Test Size**: 20% of data reserved for testing
- **Stratification**: Maintains class distribution in both sets

### 5. Class Imbalance Handling
```python
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print("scale_pos_weight:", scale_pos_weight)
```

**Purpose**: 
- Calculates the ratio of negative to positive samples
- Used in XGBoost to handle class imbalance
- Automatically adjusts the loss function

## Hyperparameter Optimization with Optuna

### Optuna Objective Function
```python
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': scale_pos_weight,
        'eval_metric': 'auc',
        'early_stopping_rounds': 30,
        'random_state': 42,
        'n_jobs': -1,
    }
```

### Hyperparameter Ranges and Explanations

#### Core Parameters
- **n_estimators**: 200-800 boosting rounds
  - More rounds = better performance but risk of overfitting
- **learning_rate**: 0.001-0.3 (logarithmic scale)
  - Lower values = more conservative boosting
  - Higher values = faster convergence but risk of overfitting

#### Tree Structure Parameters
- **max_depth**: 3-10 maximum tree depth
  - Deeper trees = more complex patterns but risk of overfitting
- **min_child_weight**: 1-10 minimum child weight
  - Higher values = more conservative splitting

#### Regularization Parameters
- **subsample**: 0.5-1.0 fraction of samples per tree
  - Lower values = more regularization
- **colsample_bytree**: 0.5-1.0 fraction of features per tree
  - Lower values = feature subsampling for regularization
- **gamma**: 0.0-5.0 minimum loss reduction for splits
  - Higher values = more conservative splitting

#### Specialized Parameters
- **scale_pos_weight**: Automatically calculated for class imbalance
- **eval_metric**: 'auc' for ROC-AUC optimization
- **early_stopping_rounds**: 30 rounds without improvement
- **n_jobs**: -1 for parallel processing

### Optimization Process
```python
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)
```

- **Trials**: 30 optimization trials
- **Direction**: Maximize ROC-AUC score
- **Progress Bar**: Built-in Optuna progress tracking
- **Early Stopping**: Prevents overfitting during optimization

## Model Training and Evaluation

### Final Model Training
```python
best_params = study.best_params
best_params.update({
    'scale_pos_weight': scale_pos_weight,
    'eval_metric': 'auc',
    'random_state': 42,
    'n_jobs': -1
})

final_model = XGBClassifier(**best_params)
final_model.fit(X_train, y_train)
```

### Prediction Generation
```python
y_pred = final_model.predict(X_test)                    # Binary predictions
y_pred_proba = final_model.predict_proba(X_test)[:, 1]  # Probability scores
```

## Performance Metrics and Visualizations

### Comprehensive Evaluation Pipeline
The script uses a progress bar to track the evaluation process:

```python
tasks = [
    "Saving classification report",
    "Plotting confusion matrix", 
    "Plotting ROC curve",
    "Plotting Precision-Recall curve",
    "Plotting feature importances",
    "Saving best hyperparameters"
]

with tqdm(total=len(tasks), desc="Saving & Plotting", ncols=100) as pbar:
    # Evaluation tasks with progress updates
```

### 1. Classification Report
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of actual occurrences of each class

### 2. Confusion Matrix
- **Purpose**: Shows actual vs predicted classifications
- **Visualization**: Heatmap with annotations
- **Colors**: Blue gradient for readability

### 3. ROC Curve
- **X-axis**: False Positive Rate (1 - Specificity)
- **Y-axis**: True Positive Rate (Sensitivity)
- **AUC**: Area Under the Curve
- **Baseline**: Random classifier diagonal line

### 4. Precision-Recall Curve
- **X-axis**: Recall
- **Y-axis**: Precision
- **PR-AUC**: Area Under Precision-Recall Curve
- **Importance**: Better metric for imbalanced datasets

### 5. Feature Importance Analysis
```python
plt.figure(figsize=(10, 6))
xgb_importance = final_model.feature_importances_
sorted_idx = np.argsort(xgb_importance)[::-1][:10]
sns.barplot(x=X.columns[sorted_idx], y=xgb_importance[sorted_idx])
plt.title('Top 10 Feature Importances')
plt.xticks(rotation=45)
plt.savefig(f'{save_dir}/feature_importances.png')
```

## Output Files

All results are saved in the `results_best/xgboost/` directory:

1. **`classification_report.txt`**: Detailed classification metrics
2. **`confusion_matrix.png`**: Confusion matrix visualization
3. **`roc_curve.png`**: ROC curve plot
4. **`precision_recall_curve.png`**: Precision-recall curve plot
5. **`feature_importances.png`**: Top 10 feature importance visualization
6. **`best_hyperparameters.txt`**: Best hyperparameters from Optuna optimization

## Key Features

### Strengths
- **Automated hyperparameter tuning**: Optuna provides efficient optimization
- **Class imbalance handling**: Automatic scale_pos_weight calculation
- **Comprehensive evaluation**: Multiple metrics and visualizations
- **Progress tracking**: Visual progress bars for long-running operations
- **Feature importance**: Built-in feature ranking for interpretability
- **Early stopping**: Prevents overfitting during training
- **Parallel processing**: Efficient use of computational resources

### XGBoost Advantages
- **High performance**: State-of-the-art gradient boosting
- **Built-in regularization**: Multiple regularization techniques
- **Handles missing values**: Robust to missing data
- **Feature importance**: Multiple importance calculation methods
- **Scalability**: Efficient implementation for large datasets
- **Flexibility**: Extensive hyperparameter options

## Performance Expectations

- **Training Time**: 
  - Hyperparameter tuning: 15-45 minutes (30 trials)
  - Final training: 2-10 minutes
- **Memory Usage**: Moderate (depends on tree depth and number of estimators)
- **Accuracy**: High performance potential with proper tuning
- **Scalability**: Excellent for large datasets

## Usage Instructions

1. **Prerequisites**: 
   - Install XGBoost: `pip install xgboost`
   - Install Optuna: `pip install optuna`
   - Ensure `creditcard.csv` is in the same directory

2. **Execution**: 
   ```bash
   python xgboost_training.py
   ```

3. **Results**: Check the `results_best/xgboost/` directory for outputs

## Troubleshooting

### Common Issues
1. **Memory issues**: Reduce `n_estimators` or `max_depth`
2. **Slow training**: Increase `learning_rate` or reduce `n_estimators`
3. **Overfitting**: Increase regularization parameters
4. **Poor performance**: Check class imbalance handling

### Optimization Tips
1. **Parameter priority**: Focus on `n_estimators`, `learning_rate`, and `max_depth` first
2. **Regularization**: Use `subsample` and `colsample_bytree` for overfitting prevention
3. **Early stopping**: Monitor validation performance to prevent overfitting
4. **Feature selection**: Use feature importance to remove irrelevant features

## Business Applications

### Fraud Detection Capabilities
- **High accuracy**: Excellent performance on imbalanced datasets
- **Real-time scoring**: Fast prediction for live transactions
- **Feature insights**: Identifies most predictive transaction features
- **Risk assessment**: Probability scores for risk-based decisions

### Operational Benefits
- **Automated monitoring**: Continuous fraud detection
- **Reduced manual review**: Better precision reduces false positives
- **Scalable solution**: Handles high-volume transaction processing
- **Cost optimization**: Minimizes fraud losses and operational costs

## Advanced Features

### Class Imbalance Handling
- **Automatic weight calculation**: `scale_pos_weight` based on class distribution
- **Loss function adjustment**: Modifies gradient boosting to handle imbalance
- **Evaluation metrics**: Focus on precision-recall for imbalanced data

### Hyperparameter Optimization
- **Bayesian optimization**: Optuna uses efficient search strategies
- **Multi-objective optimization**: Can optimize multiple metrics simultaneously
- **Pruning**: Early termination of unpromising trials

## Future Enhancements

1. **Advanced ensemble methods**:
   - Stacking with other algorithms
   - Voting classifiers
   - Blending techniques

2. **Feature engineering**:
   - Time-series features
   - Customer behavior patterns
   - Transaction sequence analysis

3. **Model improvements**:
   - Online learning for concept drift
   - Incremental updates
   - Model versioning

4. **Production deployment**:
   - Real-time API endpoints
   - Model serving infrastructure
   - Performance monitoring

5. **Advanced analytics**:
   - SHAP values for local explanations
   - Counterfactual analysis
   - Fraud pattern discovery
   - Anomaly detection integration
