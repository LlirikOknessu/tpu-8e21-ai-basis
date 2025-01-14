import argparse
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def prepare_regression_data(df):
    df_temp = pd.get_dummies(df.copy(), columns=['sex', 'smoker', 'region'], drop_first=True)
    X = df_temp.drop('charges', axis=1)
    y = df_temp['charges']
    return X, y

def plot_metrics(y_test, y_pred, output_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='purple')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'mlp_regression_actual_vs_predicted.png'), dpi=300)
    plt.close()

def plot_training_curves(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'mlp_loss_curve.png'), dpi=300)
    plt.close()

    # R2 and MAE curves
    plt.figure(figsize=(10, 5))
    plt.plot(history['r2'], label='Training R2', color='green')
    plt.plot(history['val_r2'], label='Validation R2', color='red')
    plt.title('R2 Curve')
    plt.xlabel('Epochs')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'mlp_r2_curve.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history['mae'], label='Training MAE', color='purple')
    plt.plot(history['val_mae'], label='Validation MAE', color='brown')
    plt.title('MAE Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'mlp_mae_curve.png'), dpi=300)
    plt.close()

def plot_weight_histograms(model, output_dir):
    """Save histograms of model weights."""
    os.makedirs(output_dir, exist_ok=True)
    for i, weights in enumerate(model.coefs_):
        plt.figure(figsize=(8, 6))
        plt.hist(weights.ravel(), bins=30, color='purple', alpha=0.7)
        plt.title(f'Layer {i+1} Weights Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f'mlp_layer_{i+1}_weights_histogram.png'), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train an MLP Regression model.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=False, default="models/mlp.pkl", help="Path to save the trained model.")
    parser.add_argument("--model_path", required=False, help="Alias for --output.")
    parser.add_argument("--metrics_output", required=False, default="metrics", help="Path to save metrics and plots.")
    parser.add_argument("--graphs_output", required=False, default="graphs", help="Path to save training curves and weight histograms.")
    args = parser.parse_args()
    if args.model_path:
        args.output = args.model_path

    df = pd.read_csv(args.input)
    print(f"[INFO] Data loaded: {df.shape}")

    X, y = prepare_regression_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Более простая архитектура нейронной сети
    model = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42, alpha=0.001, solver='adam', warm_start=True)
    history = {'loss': [], 'val_loss': [], 'r2': [], 'val_r2': [], 'mae': [], 'val_mae': []}

    for epoch in range(model.max_iter):
        model.partial_fit(X_train_scaled, y_train)
        train_loss = mean_squared_error(y_train, model.predict(X_train_scaled))
        val_loss = mean_squared_error(y_test, model.predict(X_test_scaled))
        train_r2 = r2_score(y_train, model.predict(X_train_scaled))
        val_r2 = r2_score(y_test, model.predict(X_test_scaled))
        train_mae = mean_absolute_error(y_train, model.predict(X_train_scaled))
        val_mae = mean_absolute_error(y_test, model.predict(X_test_scaled))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        print(f"[INFO] Epoch {epoch+1}/{model.max_iter}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, R2: {train_r2:.3f}, Val R2: {val_r2:.3f}, MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}")

    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"[INFO] R2: {r2:.3f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    metrics_path = os.path.join(args.metrics_output, 'mlp_regression_metrics.txt')
    os.makedirs(args.metrics_output, exist_ok=True)
    with open(metrics_path, 'w') as f:
        f.write(f"R2: {r2:.3f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")

    plot_metrics(y_test, y_pred, args.metrics_output)
    plot_training_curves(history, args.graphs_output)
    plot_weight_histograms(model, args.graphs_output)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(model, args.output)
    print(f"[INFO] Model saved to {args.output}")

if __name__ == "__main__":
    main()
