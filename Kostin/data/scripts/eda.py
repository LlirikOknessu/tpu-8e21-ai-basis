#!/usr/bin/env python3

"""
scripts/eda.py

Скрипт EDA: читает data/cleaned_data.csv,
генерирует графики в папку graphs/ с timestamp'ом.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
import argparse


def save_plot(filename):
    """
    Сохраняет текущий график в папку graphs/<дата_время>.
    """
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    dir_name = f"graphs/{current_time}"
    os.makedirs(dir_name, exist_ok=True)
    plt.savefig(f"{dir_name}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()


def main(input_path):
    print(f"[INFO] Reading cleaned data from {input_path}")
    df = pd.read_csv(input_path)

    # Пример EDA: числовые признаки
    num_cols = ['age', 'bmi', 'children', 'charges']

    # 1) Histograms
    sns.set_theme(color_codes=True)
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(num_cols, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
    plt.tight_layout()
    save_plot("eda_numerical_distributions.png")

    # 2) Pairplot
    sns.pairplot(df[num_cols], diag_kind='kde', corner=True)
    plt.suptitle("Pairplot (numeric features)", y=1.02)
    save_plot("eda_pairplot_numeric.png")

    # 3) Heatmap (corr matrix)
    df_copy = df.copy()
    # map smoker to 0/1 for correlation
    if 'smoker' in df_copy.columns:
        df_copy['smoker'] = df_copy['smoker'].map({'no': 0, 'yes': 1, 0: 0, 1:1})
    corr = df_copy[num_cols + (['smoker'] if 'smoker' in df_copy.columns else [])].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='YlGnBu')
    plt.title("Correlation Heatmap")
    save_plot("eda_correlation_heatmap.png")

    # 4) Boxplots для категориальных (пример: smoker, sex, region)
    cat_cols = ['smoker', 'sex', 'region']
    for cat_col in cat_cols:
        if cat_col in df.columns:
            plt.figure(figsize=(8,5))
            sns.boxplot(x=cat_col, y='charges', data=df, hue=cat_col, dodge=False)
            plt.title(f"Charges by {cat_col}")
            save_plot(f"eda_boxplot_charges_by_{cat_col}.png")

    print("[INFO] EDA completed, plots saved in graphs/ folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on cleaned data.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/cleaned_data.csv",
        help="Path to the cleaned CSV file"
    )
    args = parser.parse_args()

    main(args.input)
