import pandas as pd

df = pd.read_csv("data/train.csv")
legitimate = df[df["Class"] == 0]
legitimate.to_csv("data/legitimate.csv", index=False)
print(f"Exported {len(legitimate)} rows to data/legitimate.csv")
