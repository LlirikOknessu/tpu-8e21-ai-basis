import argparse
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def prepare_regression_data(df):
    # Учитываем новые признаки, созданные ранее
    df_temp = pd.get_dummies(df.copy(), columns=['sex', 'smoker', 'region', 'age_category'], drop_first=True)
    X = df_temp[[col for col in df_temp.columns if col not in ['charges']]]
    y = df_temp['charges']
    return X, y

def plot_metrics(y_test, y_pred, output_dir):
    """Save scatter plot of actual vs predicted values."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='orange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'linear_regression_actual_vs_predicted.png'), dpi=300)
    plt.close()

def plot_weight_distribution(model, feature_names, output_dir):
    """Save a bar plot of feature weights."""
    weights = model.coef_
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, weights, color='skyblue')
    plt.xlabel('Weight')
    plt.ylabel('Feature')
    plt.title('Feature Weights Distribution')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'linear_regression_weights.png'), dpi=300)
    plt.close()

def calculate_loss_curve(y_test, y_pred, output_dir):
    """Save a plot of loss values (RMSE) for visualization."""
    residuals = np.abs(y_test - y_pred)
    sorted_residuals = np.sort(residuals)
    cumulative_loss = np.cumsum(sorted_residuals) / np.sum(sorted_residuals)

    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0, 1, len(cumulative_loss)), cumulative_loss, label='Loss Curve')
    plt.xlabel('Proportion of Predictions')
    plt.ylabel('Cumulative Loss (Normalized)')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train a Linear Regression model.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to save the trained model.")
    parser.add_argument("--metrics_output", required=False, default="metrics", help="Path to save metrics and plots.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"[INFO] Data loaded: {df.shape}")

    X, y = prepare_regression_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"[INFO] R2: {r2:.3f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    metrics_path = os.path.join(args.metrics_output, 'linear_regression_metrics.txt')
    os.makedirs(args.metrics_output, exist_ok=True)
    with open(metrics_path, 'w') as f:
        f.write(f"R2: {r2:.3f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")

    plot_metrics(y_test, y_pred, args.metrics_output)
    plot_weight_distribution(model, X.columns, args.metrics_output)
    calculate_loss_curve(y_test, y_pred, args.metrics_output)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(model, args.output)
    print(f"[INFO] Model saved to {args.output}")

if __name__ == "__main__":
    main()