import pandas as pd
import numpy as np
import joblib
import argparse
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

parser = argparse.ArgumentParser(description="Run fraud detection inference on a CSV file.")
parser.add_argument("--input_csv",default="data/test.csv", help="Path to input CSV file")
parser.add_argument("--output", default="predictions.csv", help="Path to save predictions (default: predictions.csv)")
parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5)")
args = parser.parse_args()

# load model and scaler
model = XGBClassifier()
model.load_model("models/xgboost_fraud.json")
scaler = joblib.load("models/scaler.joblib")

# load and preprocess data
df = pd.read_csv(args.input_csv)

has_labels = "Class" in df.columns
if has_labels:
    y_true = df["Class"].copy()
    X = df.drop("Class", axis=1)
else:
    X = df.copy()

X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])

# predict
proba = model.predict_proba(X)[:, 1]
pred = (proba >= args.threshold).astype(int)

# save results
out = df.copy()
out["fraud_probability"] = proba
out["predicted_class"] = pred
out.to_csv(args.output, index=False)

print(f"Predictions saved to {args.output}")
print(f"Threshold: {args.threshold}")
print(f"Predicted fraud: {pred.sum()} / {len(pred)} ({pred.mean() * 100:.2f}%)")

if has_labels:
    from sklearn.metrics import classification_report, roc_auc_score
    print("\nClassification Report:")
    print(classification_report(y_true, pred, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_true, proba):.4f}")

    cm = confusion_matrix(y_true, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix (threshold={args.threshold})")
    plt.tight_layout()
    plot_path = args.output.replace(".csv", "_confusion_matrix.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {plot_path}")
