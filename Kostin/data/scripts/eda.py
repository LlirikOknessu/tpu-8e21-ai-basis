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

def remove_invalid_columns(data):
    """Remove columns with invalid or unexpected names."""
    invalid_cols = [col for col in data.columns if not col.isidentifier()]
    if invalid_cols:
        print(f"[INFO] Removed invalid columns: {invalid_cols}")
    return data.drop(columns=invalid_cols)

def main():
    parser = argparse.ArgumentParser(description="Perform EDA on cleaned data.")
    parser.add_argument("--input", required=True, help="Path to cleaned CSV file.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"[INFO] EDA on {args.input}, shape={df.shape}")

    # Remove invalid columns
    df = remove_invalid_columns(df)

    num_cols = ['age', 'bmi', 'children', 'charges', 'health_index', 'premium_ratio']
    cat_cols = ['sex', 'smoker', 'region', 'age_category']
    sns.set_theme(color_codes=True)

    # 1) Гистограммы числовых данных
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(num_cols, 1):
        if col in df.columns:
            plt.subplot(3, 2, i)
            sns.histplot(df[col], kde=True, bins=30, color="blue")
            plt.title(f"Distribution of {col}")
    plt.tight_layout()
    save_plot("01_numeric_distributions.png")

    # 2) Коробчатые диаграммы для числовых данных по категориям
    for cat_col in cat_cols:
        if cat_col in df.columns:
            for num_col in ['charges', 'health_index', 'premium_ratio']:
                if num_col in df.columns:
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(x=cat_col, y=num_col, data=df, palette="Set2")
                    plt.title(f"{num_col} by {cat_col}")
                    save_plot(f"02_boxplot_{num_col}_by_{cat_col}.png")

    # 3) Корреляционная тепловая карта
    if len(num_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[num_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        save_plot("03_correlation_heatmap.png")

    # 4) Парные диаграммы (pairplot)
    valid_num_cols = [col for col in num_cols if col in df.columns]
    if valid_num_cols:
        sns.pairplot(df[valid_num_cols], diag_kind="kde", corner=True)
        plt.suptitle("Pairplot of Numeric Features", y=1.02)
        save_plot("04_pairplot_numeric_features.png")

    print("[INFO] EDA completed. Graphs saved in graphs/<time>")

if __name__ == "__main__":
    main()
