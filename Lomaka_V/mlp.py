from pathlib import Path
import argparse
import os
import joblib
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

MLP_MODELS_MAPPER={'MLPRegressor': MLPRegressor}


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib',
                        required=False, help='path to linear regression prod version')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    parser.add_argument("--metrics_output", required=False, default="metrics", help="Path to save metrics and plots.")
    parser.add_argument("--graphs_output", required=False, default="graphs",
                        help="Path to save training curves and weight histograms.")
    parser.add_argument("--logdir", required=False, default="logs", help="Path to save TensorBoard logs.")
    return parser.parse_args()

def plot_metrics(y_test, y_pred, output_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='purple')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'mlp_regression_actual_vs_predictedNEWNEW.png'), dpi=300)
    plt.close()

def plot_training_curves(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'mlp_loss_curveNEWNEW.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history['r2'], label='Training R2', color='green')
    plt.plot(history['val_r2'], label='Validation R2', color='red')
    plt.title('R2 Curve')
    plt.xlabel('Epochs')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'mlp_r2_curveNEWNEW.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history['mae'], label='Training MAE', color='purple')
    plt.plot(history['val_mae'], label='Validation MAE', color='brown')
    plt.title('MAE Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'mlp_maeNEWNEW_curve.png'), dpi=300)
    plt.close()

def plot_weight_histograms(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, weights in enumerate(model.coefs_):
        plt.figure(figsize=(8, 6))
        plt.hist(weights.ravel(), bins=30, color='purple', alpha=0.7)
        plt.title(f'Layer {i+1} Weights Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f'mlp_layer_{i+1}_weightsNEW_histogram.png'), dpi=300)
        plt.close()








if __name__ == '__main__':


    args = parser_args_for_sac()


    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    parameters = params_all['mlp']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    baseline_model_path = Path(args.baseline_model)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / (args.model_name + '.joblib')

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    # # TensorBoard logger
    # log_dir = os.path.join(args.logdir, f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
    # os.makedirs(log_dir, exist_ok=True)
    # writer = tf.summary.create_file_writer(log_dir)

    log_dir = os.path.join(args.logdir, f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    writer = tf.summary.create_file_writer(log_dir)

    # Более простая архитектура нейронной сети
    model = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42, alpha=0.001, solver='adam',
                         warm_start=True)
    history = {'loss': [], 'val_loss': [], 'r2': [], 'val_r2': [], 'mae': [], 'val_mae': []}

    for epoch in range(model.max_iter):
        model.partial_fit(X_train, y_train)
        train_loss = mean_squared_error(y_train, model.predict(X_train))
        val_loss = mean_squared_error(y_test, model.predict(X_test))
        train_r2 = r2_score(y_train, model.predict(X_train))
        val_r2 = r2_score(y_test, model.predict(X_test))
        train_mae = mean_absolute_error(y_train, model.predict(X_train))
        val_mae = mean_absolute_error(y_test, model.predict(X_test))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        print(
            f"[INFO] Epoch {epoch + 1}/{model.max_iter}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, R2: {train_r2:.3f}, Val R2: {val_r2:.3f}, MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}")

        # Log metrics to TensorBoard
        with writer.as_default():
            tf.summary.scalar("Loss/Train", train_loss, step=epoch)
            tf.summary.scalar("Loss/Validation", val_loss, step=epoch)
            tf.summary.scalar("R2/Train", train_r2, step=epoch)
            tf.summary.scalar("R2/Validation", val_r2, step=epoch)
            tf.summary.scalar("MAE/Train", train_mae, step=epoch)
            tf.summary.scalar("MAE/Validation", val_mae, step=epoch)
            writer.flush()

    y_pred = model.predict(X_test)

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

    dump(model, output_model_joblib_path)

    # model = MLP_MODELS_MAPPER.get(args.model_name)()
    # model = GridSearchCV(estimator=model, param_grid=parameters[args.model_name])
    # model.fit(X_train, y_train, verbose=False)
    # model = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42, alpha=0.001, solver='adam',
    #                      warm_start=True)
    # history = {'loss': [], 'val_loss': [], 'r2': [], 'val_r2': [], 'mae': [], 'val_mae': []}
    #
    #
    #
    # for epoch in range(model.max_iter):
    #     model.partial_fit(X_train, y_train)
    #     train_loss = mean_squared_error(y_train, model.predict(X_train))
    #     val_loss = mean_squared_error(y_test, model.predict(X_test))
    #     train_r2 = r2_score(y_train, model.predict(X_train))
    #     val_r2 = r2_score(y_test, model.predict(X_test))
    #     train_mae = mean_absolute_error(y_train, model.predict(X_train))
    #     val_mae = mean_absolute_error(y_test, model.predict(X_test))
    #
    #     history['loss'].append(train_loss)
    #     history['val_loss'].append(val_loss)
    #     history['r2'].append(train_r2)
    #     history['val_r2'].append(val_r2)
    #     history['mae'].append(train_mae)
    #     history['val_mae'].append(val_mae)
    #
    #     print(f"[INFO] Epoch {epoch+1}/{model.max_iter}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, R2: {train_r2:.3f}, Val R2: {val_r2:.3f}, MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}")
    #
    #     # Log metrics to TensorBoard
    #     with writer.as_default():
    #         tf.summary.scalar("Loss/Train", train_loss, step=epoch)
    #         tf.summary.scalar("Loss/Validation", val_loss, step=epoch)
    #         tf.summary.scalar("R2/Train", train_r2, step=epoch)
    #         tf.summary.scalar("R2/Validation", val_r2, step=epoch)
    #         tf.summary.scalar("MAE/Train", train_mae, step=epoch)
    #         tf.summary.scalar("MAE/Validation", val_mae, step=epoch)
    #         writer.flush()
    #
    # y_pred = model.predict(X_test)
    #
    # r2 = r2_score(y_test, y_pred)
    # rmse = mean_squared_error(y_test, y_pred, squared=False)
    # mae = mean_absolute_error(y_test, y_pred)
    #
    # print(f"[INFO] R2: {r2:.3f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    #
    # metrics_path = os.path.join(args.metrics_output, 'mlp_regression_metrics.txt')
    # os.makedirs(args.metrics_output, exist_ok=True)
    # with open(metrics_path, 'w') as f:
    #     f.write(f"R2: {r2:.3f}\n")
    #     f.write(f"RMSE: {rmse:.2f}\n")
    #     f.write(f"MAE: {mae:.2f}\n")
    #
    # plot_metrics(y_test, y_pred, args.metrics_output)
    # plot_training_curves(history, args.graphs_output)
    # plot_weight_histograms(model, args.graphs_output)
    #
    # os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # joblib.dump(model, args.output)
    # print(f"[INFO] Model saved to {args.output}")