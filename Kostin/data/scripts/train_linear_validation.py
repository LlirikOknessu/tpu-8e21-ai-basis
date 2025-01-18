import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def prepare_regression_data(df):
    # Учитываем новые признаки, созданные ранее
    df_temp = pd.get_dummies(df.copy(), columns=['sex', 'smoker', 'region', 'age_category'], drop_first=True)
    X = df_temp[[col for col in df_temp.columns if col not in ['charges']]]
    y = df_temp['charges']
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Validate linear regression model.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--model", required=True, help="Path to the trained model file (pkl).")
    parser.add_argument("--metrics_output", required=False, default="metrics", help="Path to save validation metrics.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"[INFO] Data loaded: {df.shape}")

    X, y = prepare_regression_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load(args.model)
    print(f"[INFO] Model loaded from {args.model}")

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"[INFO] R2: {r2:.3f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    os.makedirs(args.metrics_output, exist_ok=True)
    metrics_path = os.path.join(args.metrics_output, 'linear_validation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"R2: {r2:.3f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")
    print(f"[INFO] Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
