import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def filter_iqr(data, column, multiplier=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    mask = (data[column] >= Q1 - multiplier*IQR) & (data[column] <= Q3 + multiplier*IQR)
    return data[mask]

def filter_z_score(data, column, threshold=3):
    mean_val = data[column].mean()
    std_val = data[column].std()
    z_score = (data[column] - mean_val) / std_val
    return data[abs(z_score) <= threshold]


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()



def cleaned (df_original):
    df_original = df_original.drop_duplicates()
    df_original = df_original.dropna()
    df_original = df_original.drop(columns=['student_id'])
    df_original = df_original.drop(columns=['gender'])
    return df_original



if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['data_preparation']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    for data_file in input_dir.glob('*.csv'):
        full_data = pd.read_csv(data_file)

        full_data =  full_data.drop_duplicates()
        full_data =  full_data.dropna()

        full_data =  full_data.drop(columns=['student_id'])
        full_data = full_data.drop(columns=['gender'])
        full_data = pd.get_dummies(full_data, columns=['school', 'school_setting', 'school_type', 'classroom','teaching_method', 'lunch'], drop_first=True)

        X, y = full_data.drop("posttest", axis=1), full_data['posttest']

        scaler = StandardScaler()

        y = scaler.fit_transform(pd.DataFrame(y))

        X = pd.DataFrame(X)
        y = pd.DataFrame(y)


        if 'posttest' in full_data.columns:
            full_data = filter_iqr(full_data, 'posttest', 1.5)
            full_data = filter_z_score(full_data, 'posttest', 3)

        X_train , X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=params.get('train_test_ratio'),
                                                            random_state=params.get('random_state'))
        X_train , X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          train_size=params.get('train_val_raitio'),

                                                          random_state=params.get('random_state'))

        X_full_name = output_dir / 'X_full.csv'
        y_full_name = output_dir / 'y_full.csv'
        X_train_name = output_dir / 'X_train.csv'
        y_train_name = output_dir / 'y_train.csv'
        X_test_name = output_dir / 'X_test.csv'
        y_test_name = output_dir / 'y_test.csv'
        X_val_name = output_dir / 'X_val.csv'
        y_val_name = output_dir / 'y_val.csv'

        X.to_csv(X_full_name, index=False)
        y.to_csv(y_full_name, index=False)
        X_train.to_csv(X_train_name, index=False)
        y_train.to_csv(y_train_name, index=False)
        X_test.to_csv(X_test_name, index=False)
        y_test.to_csv(y_test_name, index=False)
        X_val.to_csv(X_val_name, index=False)
        y_val.to_csv(y_val_name, index=False)





