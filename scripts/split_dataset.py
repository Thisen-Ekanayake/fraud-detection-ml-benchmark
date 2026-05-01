#!/usr/bin/env python3
"""
Split creditcard.csv into test.csv (5000 samples) and train.csv (remainder),
preserving class distribution via stratified sampling.
Outputs statistics to terminal and data/split_stats.txt.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
INPUT_FILE = f"{DATA_DIR}/creditcard.csv"
TEST_FILE = f"{DATA_DIR}/test.csv"
TRAIN_FILE = f"{DATA_DIR}/train.csv"
STATS_FILE = f"{DATA_DIR}/split_stats.txt"
TEST_SIZE = 5000
RANDOM_STATE = 42


def class_stats(df: pd.DataFrame, name: str) -> str:
    total = len(df)
    counts = df["Class"].value_counts().sort_index()
    lines = [
        f"=== {name} ===",
        f"  Total samples : {total}",
    ]
    for cls, cnt in counts.items():
        lines.append(f"  Class {cls}       : {cnt:>6}  ({cnt / total * 100:.4f}%)")
    return "\n".join(lines)


def main():
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)

    # Strip quotes from Class column if present (original CSV quotes values)
    df["Class"] = df["Class"].astype(str).str.strip('"').astype(int)

    total = len(df)
    print(f"Total rows: {total}")

    test_ratio = TEST_SIZE / total
    df_train, df_test = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["Class"],
        random_state=RANDOM_STATE,
    )

    # Verify test set size (stratified split may be off by 1)
    assert len(df_test) == TEST_SIZE, (
        f"Expected {TEST_SIZE} test samples, got {len(df_test)}"
    )

    df_train.to_csv(TRAIN_FILE, index=False)
    df_test.to_csv(TEST_FILE, index=False)

    stats_lines = [
        class_stats(df, "Original dataset"),
        "",
        class_stats(df_train, f"Train split  ({TRAIN_FILE})"),
        "",
        class_stats(df_test, f"Test split   ({TEST_FILE})"),
    ]
    stats_output = "\n".join(stats_lines)

    print()
    print(stats_output)
    print()
    print(f"Files written: {TRAIN_FILE}, {TEST_FILE}")

    with open(STATS_FILE, "w") as f:
        f.write(stats_output + "\n")
    print(f"Statistics written to {STATS_FILE}")


if __name__ == "__main__":
    main()
