import pandas as pd
import argparse
from pathlib import Path
import yaml
import os
import joblib
from math import sqrt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import dump
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

LINEAR_MODELS_MAPPER = {'Ridge': Ridge,
                        'LinearRegression': LinearRegression}


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')
    parser.add_argument("--metrics_output", required=False, default="metrics", help="Path to save metrics and plots.")
    return parser.parse_args()

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

def calculate_loss_curve(y_test, pred_baseline, output_dir):
    """Save a plot of loss values (RMSE) for visualization."""
    residuals = np.abs(y_test - pred_baseline)
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







if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / (args.model_name + '.csv')
    output_model_joblib_path = output_dir / (args.model_name + '.joblib')

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    reg = LINEAR_MODELS_MAPPER.get(args.model_name)().fit(X_train, y_train)

    y_mean = y_test.mean()
    y_pred_baseline = [y_mean] * len(y_test)

    predicted_values = np.squeeze(reg.predict(X_test))

    print(reg.score(X_test, y_test))
    print("Mean apt salary: ", y_mean)
    print("Baseline MAE: ", mean_absolute_error(y_test, y_pred_baseline))
    print("Model MAE: ", mean_absolute_error(y_test, predicted_values))

    intercept = reg.intercept_.astype(float)
    coefficients = reg.coef_.astype(float)
    intercept = pd.Series(intercept, name='intercept')
    coefficients = pd.Series(coefficients[0], name='coefficients')
    print("intercept:", intercept)
    print("list of coefficients:", coefficients)
    columns = [x for x in range(len(coefficients))]
    out_model = pd.DataFrame([coefficients, intercept])
    out_model.to_csv(output_model_path, index=False)

    dump(reg, output_model_joblib_path)

