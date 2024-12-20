import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from joblib import dump, load
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import random
from catboost import CatBoostRegressor
from catboost import Pool, cv

CATBOOST_MODELS_MAPPER = {'CatBoostRegressor': CatBoostRegressor}

# Set the best parameters that you get on training stage for all used models
CATBOOST_MODELS_BEST_PARAMETERS = {
    'CatBoostRegressor': {'depth': 10, 'learning_rate': 0.05, 'iterations': 100}}

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
    #output_model_path = output_dir / (args.model_name + '_prod.jpg') # for visualisation
    output_model_joblib_path = output_dir / (args.model_name + '_prod.joblib')

    X_train_name = input_dir / 'X_full.csv'
    y_train_name = input_dir / 'y_full.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    y_train_cols = y_train.columns

    best_params = CATBOOST_MODELS_BEST_PARAMETERS.get(args.model_name)
    reg = CATBOOST_MODELS_MAPPER.get(args.model_name)(**best_params)
    reg = reg.fit(X_train, y_train, verbose=False)

    predicted_values = np.squeeze(reg.predict(X_train))

    print(reg.score(X_train, y_train))
    print(reg.get_params)

    print("Model MAE: ", mean_absolute_error(y_train, predicted_values))

    feature_importance = reg.get_feature_importance()
    feature_names = X_train.columns

    # Display feature importance
    for name, importance in zip(feature_names, feature_importance):
        print(f"Feature: {name}, Importance: {importance:.2f}")

    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_names, color = 'b')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()
    '''
    dump(reg, output_model_joblib_path)