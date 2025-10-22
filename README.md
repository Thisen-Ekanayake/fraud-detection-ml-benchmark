# Credit Card Fraud Detection

A comprehensive machine learning project for detecting fraudulent credit card transactions using multiple algorithms and evaluation metrics.

## 📊 Dataset

The project uses the **Credit Card Fraud Detection** dataset from Kaggle, which contains:
- **284,807 transactions** (284,806 data points + header)
- **31 features** including:
  - `Time`: Seconds elapsed between each transaction and the first transaction
  - `Amount`: Transaction amount
  - `V1-V28`: Anonymized features (PCA transformed)
  - `Class`: Target variable (0 = Normal, 1 = Fraud)

### Class Distribution
- **Normal transactions**: ~99.83% (284,315 samples)
- **Fraudulent transactions**: ~0.17% (492 samples)

This severe class imbalance makes fraud detection a challenging problem requiring specialized techniques.

## 🚀 Features

- **Multiple ML Algorithms**: Logistic Regression, Neural Networks, Random Forest, and XGBoost
- **Class Imbalance Handling**: Various techniques including class weights, SMOTE, and specialized loss functions
- **Comprehensive Evaluation**: ROC curves, Precision-Recall curves, confusion matrices, and feature importance
- **Automated Results Generation**: All metrics and visualizations are automatically saved
- **GPU Support**: Neural network training supports CUDA acceleration

## 📁 Project Structure

```
Credit Card Fraud Detection/
├── creditcard.csv                    # Dataset
├── requirements.txt                  # Dependencies
├── logistic_regression_training.py  # Logistic Regression model
├── neural_network_training.py       # PyTorch Neural Network
├── random_forest_training.py       # Random Forest model
├── xgboost_training.py             # XGBoost model
├── results_1/                      # Generated results
│   ├── logistic_regression/
│   ├── neural_network/
│   ├── random_forest/
│   └── xgboost/
└── venv/                           # Virtual environment
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Credit Card Fraud Detection"
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📋 Dependencies

### Core Libraries
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning algorithms
- `xgboost` - Gradient boosting framework
- `imbalanced-learn` - Handling class imbalance
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization

### Deep Learning
- `torch` - PyTorch framework
- `torchvision` - Computer vision utilities
- `torchaudio` - Audio processing

### Development
- `jupyter` - Interactive notebooks

## 🏃‍♂️ Usage

### Running Individual Models

1. **Logistic Regression**:
   ```bash
   python logistic_regression_training.py
   ```

2. **Neural Network**:
   ```bash
   python neural_network_training.py
   ```

3. **Random Forest**:
   ```bash
   python random_forest_training.py
   ```

4. **XGBoost**:
   ```bash
   python xgboost_training.py
   ```

### Expected Outputs

Each model generates:
- **Classification Report** (`classification_report.txt`)
- **Confusion Matrix** (`confusion_matrix.png`)
- **ROC Curve** (`roc_curve.png`)
- **Precision-Recall Curve** (`precision_recall_curve.png`)
- **Feature Importance** (Random Forest & XGBoost only)

## 📈 Model Performance

Based on the generated results:

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| **Neural Network** | ~0.99+ | 0.8370 | 0.7857 | 0.8105 |
| **XGBoost** | ~0.99+ | 0.8384 | 0.8469 | 0.8426 |
| **Random Forest** | ~0.99+ | 0.8081 | 0.8163 | 0.8122 |
| **Logistic Regression** | ~0.99+ | 0.0608 | 0.9184 | 0.1141 |

### Key Insights:
- **Neural Network** and **XGBoost** show the best balance of precision and recall
- **Logistic Regression** has high recall but very low precision (many false positives)
- All models achieve excellent ROC-AUC scores (>0.99)

## 🔧 Model Configurations

### Logistic Regression
- **Solver**: L-BFGS
- **Class Weight**: Balanced
- **Max Iterations**: 1000
- **Preprocessing**: StandardScaler on all features

### Neural Network (PyTorch)
- **Architecture**: 3-layer MLP (input → 64 → 32 → 1)
- **Activation**: ReLU + Sigmoid
- **Dropout**: 0.3 (layer 1), 0.2 (layer 2)
- **Loss**: Binary Cross-Entropy with positive class weighting
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 50
- **Batch Size**: 2048

### Random Forest
- **Estimators**: 300
- **Max Depth**: 10
- **Class Weight**: Balanced
- **Parallel Processing**: Enabled (n_jobs=-1)
- **Preprocessing**: StandardScaler on Time and Amount features

### XGBoost
- **Estimators**: 400
- **Learning Rate**: 0.05
- **Max Depth**: 5
- **Subsample**: 0.8
- **Column Sampling**: 0.8
- **Scale Pos Weight**: Automatically calculated
- **Evaluation Metric**: AUC

## 📊 Evaluation Metrics

The project uses multiple evaluation metrics suitable for imbalanced datasets:

- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **Precision-Recall AUC**: Area under the Precision-Recall curve
- **Confusion Matrix**: True/False Positives and Negatives
- **Classification Report**: Precision, Recall, F1-Score for each class
- **Feature Importance**: For tree-based models

## 🎯 Key Features

### Class Imbalance Handling
- **Logistic Regression**: `class_weight='balanced'`
- **Neural Network**: Positive class weighting in loss function
- **Random Forest**: `class_weight='balanced'`
- **XGBoost**: `scale_pos_weight` parameter

### Data Preprocessing
- **Feature Scaling**: StandardScaler for numerical features
- **Stratified Split**: Maintains class distribution in train/test sets
- **Feature Engineering**: Time and Amount features are specifically scaled

### Visualization
- **Heatmaps**: Confusion matrices with annotations
- **Curves**: ROC and Precision-Recall curves with AUC scores
- **Feature Importance**: Bar plots for top 10 most important features
- **Consistent Styling**: All plots use seaborn for professional appearance

## 🔍 Results Analysis

The `results_1/` directory contains comprehensive outputs for each model:

```
results_1/
├── logistic_regression/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   └── roc_curve.png
├── neural_network/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── precision-recall_curve.png
│   └── roc_curve.png
├── random_forest/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── precision-recall_curve.png
│   └── roc_curve.png
└── xgboost/
    ├── classification_report.txt
    ├── confusion_matrix.png
    ├── feature_importances.png
    ├── precision_recall_curve.png
    └── roc_curve.png
```

## 🚨 Important Notes

1. **Dataset**: Ensure `creditcard.csv` is in the project root directory
2. **GPU Support**: Neural network training will automatically use GPU if available
3. **Memory Requirements**: Neural network training may require significant RAM for large batch sizes
4. **Reproducibility**: All models use `random_state=42` for consistent results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🙏 Acknowledgments

- **Dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle
- **Libraries**: scikit-learn, PyTorch, XGBoost, and the Python data science ecosystem
