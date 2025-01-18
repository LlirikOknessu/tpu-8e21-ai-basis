import argparse
import os
import joblib
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def prepare_regression_data(df):
    # Учитываем новые признаки, созданные ранее
    df_temp = pd.get_dummies(df.copy(), columns=['sex', 'smoker', 'age_category'], drop_first=True)
    # Добавляем все новые и оригинальные признаки, исключая 'region'
    X = df_temp[[col for col in df_temp.columns if col not in ['charges', 'region']]]
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

def plot_feature_importance(model, feature_names, output_dir):
    """Save feature importance plot."""
    feature_importances = model.get_feature_importance()
    sorted_idx = feature_importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importances)), feature_importances[sorted_idx], align='center', color='skyblue')
    plt.yticks(range(len(feature_importances)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('CatBoost Feature Importance')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'catboost_feature_importance.png'), dpi=300)
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
    plot_feature_importance(model, X.columns, args.metrics_output)

    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    joblib.dump(model, args.model_output)
    print(f"[INFO] Model saved to {args.model_output}")

if __name__ == "__main__":
    main()