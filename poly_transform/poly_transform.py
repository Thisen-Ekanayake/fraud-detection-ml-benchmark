import pandas as pd
import numpy as np

df = pd.read_csv("creditcard.csv")

feature_cols = [col for col in df.columns if col != "Class"]

transformed = df.copy()

for col in feature_cols:
    vals = df[col].values
    poly_sum = sum(vals ** i for i in range(1, 6))
    transformed[col] = poly_sum / 5

transformed.to_csv("creditcard_poly.csv", index=False)
print("Saved to creditcard_poly.csv")
