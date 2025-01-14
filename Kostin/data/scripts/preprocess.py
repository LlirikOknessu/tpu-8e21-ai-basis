#!/usr/bin/env python3

"""
scripts/preprocess.py

Скрипт предобработки (очистки) данных из raw/insurance.csv.
Убираем дубликаты, пропуски, выбросы.
Сохраняем результат в data/cleaned_data.csv.
"""

import pandas as pd
import numpy as np
import argparse
import os


def filter_iqr(data, column, multiplier=1.5):
    """
    Фильтрует выбросы по IQR, умноженному на multiplier.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    mask = (data[column] >= Q1 - multiplier * IQR) & (data[column] <= Q3 + multiplier * IQR)
    return data[mask]


def filter_z_score(data, column, threshold=3):
    """
    Фильтрует выбросы по Z-score (порог threshold).
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    z_score = (data[column] - mean_val) / std_val
    return data[abs(z_score) <= threshold]


def main(input_path, output_path):
    print(f"[INFO] Reading raw data from {input_path}")
    df = pd.read_csv(input_path)

    # Удаляем дубли и пропуски
    print("[INFO] Dropping duplicates and NaNs")
    df = df.drop_duplicates()
    df = df.dropna()

    # Фильтрация выбросов по charges
    print("[INFO] Filtering outliers (IQR + Z-score) for 'charges'")
    df = filter_iqr(df, 'charges', multiplier=1.5)
    df = filter_z_score(df, 'charges', threshold=3)

    # Создаем выходную директорию, если отсутствует
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Cleaned data saved to {output_path} with shape {df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess insurance dataset.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="raw/insurance.csv",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/cleaned_data.csv",
        help="Path to the cleaned output CSV file"
    )
    args = parser.parse_args()

    main(args.input, args.output)
