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
MLP_MODELS_BEST_PARAMETERS = {
    'MLPRegressor': {'hidden_layer_sizes': 64, 'max_iter': 500, 'random_state': 42, 'alpha': 0.001, 'solver': 'adam'}}

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')

    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')

    return parser.parse_args()



if __name__ == '__main__':

    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / (args.model_name + '_prod.csv')
    output_model_joblib_path = output_dir / (args.model_name + '_prod.joblib')

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'


    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    y_train_cols = y_train.columns

    best_params = MLP_MODELS_BEST_PARAMETERS.get(args.model_name)
    model = MLP_MODELS_MAPPER.get(args.model_name)(**best_params)
    #model = model.fit(X_train, y_train, verbose=False)
    model.partial_fit(X_train, y_train)
    predicted_values = np.squeeze(model.predict(X_train))


    print("R2:", model.score(X_train, y_train))
    dump(model, output_model_joblib_path)