import pandas as pd

df = pd.read_csv("creditcard.csv")

fraud = df[df["Class"] == 1].sample(n=10, random_state=42)
legit = df[df["Class"] == 0].sample(n=40, random_state=42)

sample = pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)

sample.to_csv("data.csv", index=False)

print(f"Exported data.csv: {len(sample)} rows ({len(fraud)} fraud, {len(legit)} legitimate)")
