import argparse
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
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
    plt.scatter(y_test, y_pred, alpha=0.5, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'decision_tree_actual_vs_predicted.png'), dpi=300)
    plt.close()

def plot_training_loss(model, X_train, y_train, output_dir):
    """Save a plot of training loss during the decision tree training process."""
    train_loss = []
    for depth in range(1, model.get_depth() + 1):
        temp_model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        temp_model.fit(X_train, y_train)
        y_pred_train = temp_model.predict(X_train)
        loss = mean_squared_error(y_train, y_pred_train)
        train_loss.append(loss)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', color='blue')
    plt.xlabel('Tree Depth')
    plt.ylabel('Training Loss (MSE)')
    plt.title('Training Loss vs Tree Depth')
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'decision_tree_training_loss.png'), dpi=300)
    plt.close()

def plot_decision_tree(model, feature_names, output_dir):
    """Save the first few levels of the decision tree as an image."""
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, max_depth=3, filled=True, fontsize=10)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'decision_tree_structure.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train a Decision Tree regression model.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--model_output", required=False, default="models/decision_tree.pkl", help="Path to save the trained model.")
    parser.add_argument("--out_model", required=False, help="Alias for --model_output.")
    parser.add_argument("--metrics_output", required=False, default="metrics", help="Path to save metrics and plots.")
    args = parser.parse_args()

    if args.out_model:
        args.model_output = args.out_model

    df = pd.read_csv(args.input)
    print(f"[INFO] Data loaded: {df.shape}")

    X, y = prepare_regression_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"[INFO] R2: {r2:.3f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    metrics_path = os.path.join(args.metrics_output, 'decision_tree_metrics.txt')
    os.makedirs(args.metrics_output, exist_ok=True)
    with open(metrics_path, 'w') as f:
        f.write(f"R2: {r2:.3f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")

    plot_metrics(y_test, y_pred, args.metrics_output)
    plot_training_loss(model, X_train, y_train, args.metrics_output)
    plot_decision_tree(model, X.columns, args.metrics_output)

    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    joblib.dump(model, args.model_output)
    print(f"[INFO] Model saved to {args.model_output}")

if __name__ == "__main__":
    main()
