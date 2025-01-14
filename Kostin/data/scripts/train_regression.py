#!/usr/bin/env python3

"""
scripts/train_regression.py

Скрипт обучения модели (регрессия) для предсказания 'charges'.
Можно обучить несколько моделей и сохранить одну (или все).
"""

import pandas as pd
import numpy as np
import argparse
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor


def prepare_regression_data(df):
    """
    Преобразует категориальные признаки в One-Hot (drop_first=True),
    возвращает X, y (charges).
    """
    df_temp = df.copy()
    df_temp = pd.get_dummies(df_temp, columns=['sex','smoker','region'], drop_first=True)
    X = df_temp.drop('charges', axis=1)
    y = df_temp['charges']
    return X, y


def main(input_path, output_path):
    print(f"[INFO] Reading cleaned data from {input_path}")
    df = pd.read_csv(input_path)

    print("[INFO] Preparing features for regression.")
    X, y = prepare_regression_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Обучаем несколько моделей (пример)
    print("[INFO] Training multiple regression models.")

    # 1) LinearRegression
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred_lin = linreg.predict(X_test)
    r2_lin = r2_score(y_test, y_pred_lin)
    rmse_lin = mean_squared_error(y_test, y_pred_lin, squared=False)
    print(f"LinearRegression => R2: {r2_lin:.3f}, RMSE: {rmse_lin:.2f}")

    # 2) DecisionTreeRegressor
    dt_reg = DecisionTreeRegressor(random_state=42, max_depth=5)
    dt_reg.fit(X_train, y_train)
    y_pred_dt = dt_reg.predict(X_test)
    r2_dt = r2_score(y_test, y_pred_dt)
    rmse_dt = mean_squared_error(y_test, y_pred_dt, squared=False)
    print(f"DecisionTreeRegressor => R2: {r2_dt:.3f}, RMSE: {rmse_dt:.2f}")

    # 3) CatBoostRegressor
    cbr = CatBoostRegressor(verbose=0, random_state=42)
    cbr.fit(X_train, y_train)
    y_pred_cbr = cbr.predict(X_test)
    r2_cbr = r2_score(y_test, y_pred_cbr)
    rmse_cbr = mean_squared_error(y_test, y_pred_cbr, squared=False)
    print(f"CatBoostRegressor => R2: {r2_cbr:.3f}, RMSE: {rmse_cbr:.2f}")

    # 4) MLPRegressor (с масштабированием)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp_reg = MLPRegressor(hidden_layer_sizes=(64,64),
                           max_iter=500,
                           random_state=42)
    mlp_reg.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp_reg.predict(X_test_scaled)
    r2_mlp = r2_score(y_test, y_pred_mlp)
    rmse_mlp = mean_squared_error(y_test, y_pred_mlp, squared=False)
    print(f"MLPRegressor => R2: {r2_mlp:.3f}, RMSE: {rmse_mlp:.2f}")

    # Допустим, сохраняем только одну модель, например CatBoostRegressor, как финальную.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(cbr, output_path)
    print(f"[INFO] Final model (CatBoostRegressor) saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train regression model for insurance charges.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/cleaned_data.csv",
        help="Path to cleaned CSV file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/model_reg.pkl",
        help="Path to save the trained regression model."
    )
    args = parser.parse_args()

    main(args.input, args.output)
