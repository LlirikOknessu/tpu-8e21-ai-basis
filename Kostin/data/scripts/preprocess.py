#!/usr/bin/env python3
"""
scripts/preprocess.py
Считывает --input (raw CSV), удаляет дубликаты/пропуски/выбросы,
сохраняет в --output (cleaned_data.csv).
"""

import argparse
import os
import pandas as pd

def filter_iqr(data, column, multiplier=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    mask = (data[column] >= Q1 - multiplier*IQR) & (data[column] <= Q3 + multiplier*IQR)
    return data[mask]

def filter_z_score(data, column, threshold=3):
    mean_val = data[column].mean()
    std_val = data[column].std()
    z_score = (data[column] - mean_val) / std_val
    return data[abs(z_score) <= threshold]

def main():
    parser = argparse.ArgumentParser(description="Preprocess insurance dataset.")
    parser.add_argument("--input", required=True, help="Path to raw CSV file (e.g. raw/insurance.csv)")
    parser.add_argument("--output", required=True, help="Path to output cleaned CSV file (e.g. data/cleaned_data.csv)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.drop_duplicates().dropna()

    # Фильтрация выбросов по 'charges'
    if 'charges' in df.columns:
        df = filter_iqr(df, 'charges', 1.5)
        df = filter_z_score(df, 'charges', 3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[INFO] Preprocessed data saved to {args.output}, shape={df.shape}")

if __name__ == "__main__":
    main()
