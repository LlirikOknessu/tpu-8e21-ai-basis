#!/usr/bin/env python3
"""
scripts/eda.py
Делает расширенный EDA, используя --input data/cleaned_data.csv
и сохраняет графики в папку graphs/<date_time>.
"""

import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def save_plot(filename):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    dir_name = f"graphs/{current_time}"
    os.makedirs(dir_name, exist_ok=True)
    plt.savefig(f"{dir_name}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Perform EDA on cleaned data.")
    parser.add_argument("--input", required=True, help="Path to cleaned CSV file.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"[INFO] EDA on {args.input}, shape={df.shape}")

    num_cols = ['age','bmi','children','charges']
    cat_cols = ['sex','smoker','region']
    sns.set_theme(color_codes=True)

    # 1) Гистограммы
    plt.figure(figsize=(12,8))
    for i, col in enumerate(num_cols, 1):
        plt.subplot(2,2,i)
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
    plt.tight_layout()
    save_plot("01_eda_numerical_distributions.png")

    # 2) Pairplot
    sns.pairplot(df[num_cols], diag_kind='kde', corner=True)
    plt.suptitle("Pairplot (numeric features)", y=1.02)
    save_plot("02_pairplot_numeric.png")

    # 3) Heatmap (corr)
    df_corr = df.copy()
    if 'smoker' in df_corr.columns:
        df_corr['smoker'] = df_corr['smoker'].map({'no':0,'yes':1})
    corr_cols = [c for c in num_cols if c in df_corr.columns]
    if 'smoker' in df_corr.columns:
        corr_cols.append('smoker')
    if corr_cols:
        plt.figure(figsize=(8,6))
        sns.heatmap(df_corr[corr_cols].corr(), annot=True, cmap='YlGnBu')
        plt.title("Correlation Heatmap")
        save_plot("03_correlation_heatmap.png")

    # 4) Boxplots
    for cat_col in cat_cols:
        if cat_col in df.columns and 'charges' in df.columns:
            plt.figure(figsize=(8,5))
            sns.boxplot(x=cat_col, y='charges', data=df, hue=cat_col, dodge=False)
            plt.title(f"Charges by {cat_col}")
            save_plot(f"04_boxplot_charges_by_{cat_col}.png")

    print("[INFO] EDA completed. Graphs saved in graphs/<time>")

if __name__ == "__main__":
    main()
