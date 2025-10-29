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
├── logistic_regression_training.py   # Logistic Regression model
├── neural_network_training.py       # PyTorch Neural Network
├── random_forest_training.py        # Random Forest model
├── xgboost_training.py              # XGBoost model
├── results_best/                    # Generated best results
│   ├── logistic_regression/
│   ├── neural_network/
│   ├── random_forest/
│   └── xgboost/
└── venv/                           # Virtual environment
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Thisen-Ekanayake/fraud-detection-ml-benchmark.git
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

Based on the best results:

| Model | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| **XGBoost** | **0.8632** | **0.8367** | **0.8497** | **0.9995** |
| **Random Forest** | 0.8081 | 0.8163 | 0.8122 | 0.9994 |
| **Neural Network** | 0.7979 | 0.7653 | 0.7812 | 0.9993 |
| **Logistic Regression** | 0.8310 | 0.6020 | 0.6982 | 0.9991 |

### Why XGBoost is Best for Tabular Imbalanced Datasets:

**XGBoost** emerges as the superior choice for this credit card fraud detection task due to several key advantages:

1. **Superior Performance**: Highest F1-score (0.8497) and precision (0.8632) among all algorithms
2. **Built-in Imbalance Handling**: `scale_pos_weight` parameter automatically adjusts for class imbalance
3. **Robust Feature Learning**: Gradient boosting excels at capturing complex patterns in tabular data
4. **Regularization**: Built-in L1/L2 regularization prevents overfitting
5. **Efficiency**: Fast training and prediction, suitable for production environments
6. **Interpretability**: Feature importance scores provide insights into fraud indicators

### Key Insights:
- **XGBoost** achieves the best balance of precision and recall for fraud detection
- **Random Forest** shows competitive performance with good stability
- **Neural Network** performs well but requires more computational resources
- **Logistic Regression** has high precision but lower recall, missing more fraud cases


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


## 🚨 Important Notes

1. **Dataset**: Ensure `creditcard.csv` is in the project root directory
2. **GPU Support**: Neural network training will automatically use GPU if available
3. **Memory Requirements**: Neural network training may require significant RAM for large batch sizes
4. **Reproducibility**: All models use `random_state=42` for consistent results
5. **Class Imbalance**: This dataset has severe class imbalance (0.17% fraud cases) - consider this when interpreting results
6. **Feature Engineering**: The V1-V28 features are PCA-transformed and anonymized for privacy
7. **Evaluation**: Focus on precision-recall metrics rather than accuracy due to class imbalance

## 🤝 Contributing

We welcome contributions to improve this fraud detection project! Here's how you can help:

1. **Fork the repository** and create a feature branch
2. **Add new algorithms** (e.g., LightGBM, CatBoost, SVM)
3. **Improve preprocessing** techniques for better feature engineering
4. **Enhance evaluation** with additional metrics (e.g., Matthews Correlation Coefficient)
5. **Add ensemble methods** combining multiple algorithms
6. **Optimize hyperparameters** using advanced techniques (e.g., Optuna, Hyperopt)
7. **Add tests** for model validation and reproducibility
8. **Submit a pull request** with detailed description of changes

### Areas for Improvement:
- **Feature Engineering**: Create new features from Time and Amount
- **Advanced Sampling**: Implement ADASYN, BorderlineSMOTE, or other techniques
- **Model Interpretability**: Add SHAP values or LIME explanations
- **Real-time Detection**: Implement streaming prediction capabilities
- **Cross-validation**: Add stratified k-fold validation for robust evaluation

## 🙏 Acknowledgments

### Dataset & Research
- **Dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle
- **Data Privacy**: Features V1-V28 are PCA-transformed for confidentiality

### Libraries & Frameworks
- **scikit-learn**: Core machine learning algorithms and utilities
- **PyTorch**: Deep learning framework for neural network implementation
- **XGBoost**: Gradient boosting framework for superior tabular data performance
- **imbalanced-learn**: Specialized tools for handling class imbalance
- **matplotlib & seaborn**: Data visualization and plotting
- **pandas & numpy**: Data manipulation and numerical computing

### Community
- **Kaggle Community**: For dataset availability and ML discussions
- **Python Data Science Ecosystem**: Open-source tools and libraries
- **Machine Learning Community**: Research papers and best practices for fraud detection

### Special Thanks
- Contributors who helped improve model performance
- Reviewers who provided feedback on evaluation metrics
- The open-source community for maintaining these excellent tools
