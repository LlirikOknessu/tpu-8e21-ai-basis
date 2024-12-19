import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from joblib import dump, load
import random
from catboost import CatBoostRegressor
from catboost import Pool, cv


CATBOOST_MODELS_MAPPER = {'CatBoostRegressor': CatBoostRegressor}

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
    #parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
    #                    help='file with dvc stage params')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args_for_sac()

    #with open(args.params, 'r') as f:
    #    params_all = yaml.safe_load(f)
    #params = params_all['decision_tree']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / (args.model_name + '.csv')
    output_model_joblib_path = output_dir / (args.model_name + '.joblib')

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    cat = CATBOOST_MODELS_MAPPER.get(args.model_name)().fit(X_train, y_train, verbose=False, plot=True)

    params = {"iterations": 100,
              "depth": 2,
              "loss_function": "RMSE",
              "verbose": False}
    cv_dataset = Pool(data=X_train,
                      label=y_train)
    scores = cv(cv_dataset,
                params,
                fold_count=2,
                plot="True")

    grid = {'learning_rate': [0.03, 0.1],
            'depth': [4, 6, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9]}

    grid_search_result = cat.grid_search(grid, X=X_train, y=y_train, plot=True)
    cat.plot_tree(tree_idx=0)