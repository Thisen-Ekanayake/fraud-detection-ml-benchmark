import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve,
    auc, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ========================================
# load and preprocess the data
# ========================================

df = pd.read_csv('creditcard.csv')

X = df.drop('Class', axis=1)
y = df['Class']

# scale all features for logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ========================================
# handle imbalance using class weights
# ========================================

model = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced', # balance fraud vs non-fraud
    random_state=42
)

# ========================================
# train model
# ========================================

model.fit(X_train, y_train)

# ========================================
# predictions
# ========================================
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]