#!/usr/bin/env python3

import argparse
import os
import joblib
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def prepare_regression_data(df):
    df_temp = pd.get_dummies(df.copy(), columns=['sex','smoker','region'], drop_first=True)
    X = df_temp.drop('charges', axis=1)
    y = df_temp['charges']
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train CatBoost (regression).")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X, y = prepare_regression_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model = CatBoostRegressor(verbose=0, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"[TRAIN] CatBoostRegressor => R2={r2:.3f}, RMSE={rmse:.2f}")

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(model, args.model_path)
    print(f"[INFO] CatBoost model saved to {args.model_path}")

if __name__ == "__main__":
    main()
