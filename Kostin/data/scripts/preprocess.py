import argparse
import os
import pandas as pd
import numpy as np

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

def filter_distance_to_line(data, column_actual, column_predicted, max_distance):
    """Remove rows where the distance to the line Actual = Predicted exceeds a threshold."""
    distance = np.abs(data[column_actual] - data[column_predicted]) / np.sqrt(2)
    return data[distance <= max_distance]

def filter_charges_threshold(data, column, max_value):
    """Remove rows where charges exceed a maximum value."""
    return data[data[column] <= max_value]

def remove_invalid_columns(data, invalid_keywords):
    """Remove columns that contain invalid keywords in their names."""
    columns_to_remove = [col for col in data.columns if any(keyword in col for keyword in invalid_keywords)]
    return data.drop(columns=columns_to_remove), columns_to_remove

def main():
    parser = argparse.ArgumentParser(description="Preprocess insurance dataset.")
    parser.add_argument("--input", required=True, help="Path to raw CSV file (e.g. raw/insurance.csv)")
    parser.add_argument("--output", required=True, help="Path to output cleaned CSV file (e.g. data/cleaned_data.csv)")
    parser.add_argument("--predicted", required=False, help="Optional column for predicted values to filter extreme differences.")
    parser.add_argument("--max_distance", type=float, required=False, default=5000, help="Maximum distance from Actual = Predicted line to keep rows (default: 5000).")
    parser.add_argument("--max_charges", type=float, required=False, default=None, help="Maximum allowed value for charges (default: None).")
    parser.add_argument("--invalid_keywords", nargs="*", required=False, default=["1639"], help="Keywords to identify invalid columns.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.drop_duplicates().dropna()

    df, removed_columns = remove_invalid_columns(df, args.invalid_keywords)
    print(f"[INFO] Removed columns: {removed_columns}")

    # Фильтрация выбросов по 'charges'
    if 'charges' in df.columns:
        df = filter_iqr(df, 'charges', 1.5)
        df = filter_z_score(df, 'charges', 3)

    # Ограничение максимального значения charges
    if args.max_charges is not None:
        df = filter_charges_threshold(df, 'charges', args.max_charges)

    # Фильтрация на основе расстояния до линии Actual = Predicted
    if args.predicted and args.predicted in df.columns:
        df = filter_distance_to_line(df, 'charges', args.predicted, args.max_distance)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[INFO] Preprocessed data saved to {args.output}, shape={df.shape}")

if __name__ == "__main__":
    main()
