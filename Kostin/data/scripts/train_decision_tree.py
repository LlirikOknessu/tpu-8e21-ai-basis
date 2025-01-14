#!/usr/bin/env python3

import argparse
import os
import joblib
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def prepare_regression_data(df):
    df_temp = pd.get_dummies(df.copy(), columns=['sex','smoker','region'], drop_first=True)
    X = df_temp.drop('charges', axis=1)
    y = df_temp['charges']
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train decision tree (regression).")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_model", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X, y = prepare_regression_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model = DecisionTreeRegressor(random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"[TRAIN] DecisionTree => R2={r2:.3f}, RMSE={rmse:.2f}")

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    joblib.dump(model, args.out_model)
    print(f"[INFO] Decision tree saved to {args.out_model}")

if __name__ == "__main__":
    main()
