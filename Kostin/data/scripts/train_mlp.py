#!/usr/bin/env python3

import argparse
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

def prepare_regression_data(df):
    df_temp = pd.get_dummies(df.copy(), columns=['sex','smoker','region'], drop_first=True)
    X = df_temp.drop('charges', axis=1)
    y = df_temp['charges']
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train MLP Regressor.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X, y = prepare_regression_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp = MLPRegressor(hidden_layer_sizes=(64,64), max_iter=500, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"[TRAIN] MLPRegressor => R2={r2:.3f}, RMSE={rmse:.2f}")

    # Сохраняем (mlp, scaler)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump((mlp, scaler), args.model_path)
    print(f"[INFO] MLP (model+scaler) saved to {args.model_path}")

if __name__ == "__main__":
    main()
