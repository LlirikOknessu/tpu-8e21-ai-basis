#!/usr/bin/env python3

"""
scripts/train_classification.py

Скрипт обучения модели классификации (pred. 'smoker').
Можно обучить несколько моделей и сохранить одну как итоговую.
"""

import pandas as pd
import numpy as np
import argparse
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier


def prepare_classification_data(df):
    """
    Преобразуем 'smoker' -> 0/1, выносим в y.
    Остальные категориальные (sex, region) в One-Hot.
    Оставляем 'charges', 'age', 'bmi' и т.д. как фичи.
    """
    df_temp = df.copy()
    df_temp['smoker'] = df_temp['smoker'].map({'no': 0, 'yes': 1})
    y = df_temp['smoker']
    df_temp = df_temp.drop('smoker', axis=1)

    df_temp = pd.get_dummies(df_temp, columns=['sex','region'], drop_first=True)
    X = df_temp
    return X, y


def main(input_path, output_path):
    print(f"[INFO] Reading cleaned data from {input_path}")
    df = pd.read_csv(input_path)

    print("[INFO] Preparing features for classification.")
    X, y = prepare_classification_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Обучаем несколько моделей (пример)
    print("[INFO] Training multiple classification models.")

    # 1) LogisticRegression
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train)
    y_pred_log = logreg.predict(X_test)
    acc_log = accuracy_score(y_test, y_pred_log)
    f1_log = f1_score(y_test, y_pred_log)
    print(f"LogisticRegression => Accuracy: {acc_log:.3f}, F1: {f1_log:.3f}")

    # 2) CatBoostClassifier
    cbc = CatBoostClassifier(verbose=0, random_state=42)
    cbc.fit(X_train, y_train)
    y_pred_cbc = cbc.predict(X_test)
    acc_cbc = accuracy_score(y_test, y_pred_cbc)
    f1_cbc = f1_score(y_test, y_pred_cbc)
    print(f"CatBoostClassifier => Accuracy: {acc_cbc:.3f}, F1: {f1_cbc:.3f}")

    # 3) MLPClassifier (с масштабированием)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp_clf = MLPClassifier(hidden_layer_sizes=(64,64),
                            max_iter=500,
                            random_state=42)
    mlp_clf.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp_clf.predict(X_test_scaled)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    f1_mlp = f1_score(y_test, y_pred_mlp)
    print(f"MLPClassifier => Accuracy: {acc_mlp:.3f}, F1: {f1_mlp:.3f}")

    # Допустим, сохраняем CatBoostClassifier как финальный
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(cbc, output_path)
    print(f"[INFO] Final model (CatBoostClassifier) saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classification model for smoker prediction.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/cleaned_data.csv",
        help="Path to cleaned CSV file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/model_clf.pkl",
        help="Path to save the trained classification model."
    )
    args = parser.parse_args()

    main(args.input, args.output)
