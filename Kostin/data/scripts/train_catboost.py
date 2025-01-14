import argparse
import os
import joblib
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def prepare_regression_data(df):
    df_temp = pd.get_dummies(df.copy(), columns=['sex', 'smoker'], drop_first=True)
    # 'age' is explicitly included as a feature, and 'region' is removed
    X = df_temp[['age'] + [col for col in df_temp.columns if col not in ['age', 'charges', 'region']]]
    y = df_temp['charges']
    return X, y

def plot_metrics(y_test, y_pred, output_dir):
    """Save scatter plot of actual vs predicted values."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'catboost_actual_vs_predicted.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train a CatBoost regression model.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--model_output", required=False, default="models/catboost.pkl", help="Path to save the trained model.")
    parser.add_argument("--model_path", required=False, help="Alias for --model_output.")
    parser.add_argument("--metrics_output", required=False, default="metrics", help="Path to save metrics and plots.")
    args = parser.parse_args()

    if args.model_path:
        args.model_output = args.model_path

    df = pd.read_csv(args.input)
    print(f"[INFO] Data loaded: {df.shape}")

    X, y = prepare_regression_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(verbose=0, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"[INFO] R2: {r2:.3f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    metrics_path = os.path.join(args.metrics_output, 'catboost_metrics.txt')
    os.makedirs(args.metrics_output, exist_ok=True)
    with open(metrics_path, 'w') as f:
        f.write(f"R2: {r2:.3f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")

    plot_metrics(y_test, y_pred, args.metrics_output)
    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    joblib.dump(model, args.model_output)
    print(f"[INFO] Model saved to {args.model_output}")

if __name__ == "__main__":
    main()
