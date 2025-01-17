import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from pathlib import Path
import pandas as pd

import datetime
import shutil

import argparse
import yaml
import numpy as np
from joblib import dump, load
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import random

@tf.keras.utils.register_keras_serializable()
class NeuralNet(Model):
    def __init__(self, neurons_cnt_input = 8, dense_number = 2, neurons_cnt_d1 = 128, neurons_cnt_d2 = 128, **kwargs):
        super(NeuralNet, self).__init__(**kwargs)
        self.neurons_cnt_input = neurons_cnt_input  # Сохраняем значение параметра для конфигурации
        self.dense_number = dense_number
        self.neurons_cnt_d1 = neurons_cnt_d1
        self.neurons_cnt_d2 = neurons_cnt_d2
        self.d_in = Dense(neurons_cnt_input, activation='relu')
        self.d1 = Dense(neurons_cnt_d1, activation='relu')
        self.d2 = Dense(neurons_cnt_d2, activation='relu')
        self.d_out = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.d_in(x)
        x = self.d1(x)
        if (self.dense_number == 2):
            x = self.d2(x)
        return self.d_out(x)

    def get_config(self):
        # Возвращаем параметры модели, включая кастомные
        config = super(NeuralNet, self).get_config()
        if (self.dense_number == 2):
            config.update({
                "neurons_cnt_input": self.neurons_cnt_input,  #  Добавляем кастомный параметр в конфигурацию
                "dense_number": self.dense_number,
                "neurons_cnt_d1": self.neurons_cnt_d1,
                "neurons_cnt_d2": self.neurons_cnt_d2
            })
        else:
            config.update({
                "neurons_cnt_input": self.neurons_cnt_input,  # Добавляем кастомный параметр в конфигурацию
                "dense_number": self.dense_number,
                "neurons_cnt_d1": self.neurons_cnt_d1
            })
        return config

    @classmethod
    def from_config(cls, config):
        # Создаём экземпляр класса из конфигурации
        return cls(**config)


NeuralNet_MODELS_MAPPER = {'NeuralNet': NeuralNet}

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--input_model', '-im', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib',
                        required=False, help='path to linear regression prod version')
    parser.add_argument('--logs_dir', '-lg', type=str, default='data/logs',
                        required=False, help='path to save logs folder')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    input_model = Path(args.input_model)
    baseline_model_path = Path(args.baseline_model)
    logs_path = Path(args.logs_dir)
    '''
    if logs_path.exists():
        shutil.rmtree(logs_path)
    '''
    logs_path.mkdir(exist_ok=True,parents=True)

    baseline_model_path = Path(args.baseline_model)

    X_val_name = input_dir / 'X_val.csv'
    y_val_name = input_dir / 'y_val.csv'

    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)

    loaded_model = keras.models.load_model(input_model, NeuralNet_MODELS_MAPPER)#.get(args.model_name)

    predicted_values = loaded_model(X_val, training=False)

    baseline_model = load(baseline_model_path)
    y_pred_baseline = np.squeeze(baseline_model.predict(X_val))

    print("Baseline MAE: ", mean_absolute_error(y_val, y_pred_baseline))
    print("Model MAE: ", mean_absolute_error(y_val, predicted_values))

    test_accuracy = tf.keras.metrics.R2Score(class_aggregation='uniform_average', num_regressors=0,
                                             name='test_r2_score', dtype=None)

    test_accuracy.update_state(y_val, predicted_values)
    result = test_accuracy.result()
    print("Baseline R2: ", baseline_model.score(X_val, y_val))
    print("Model R2: ", result)





    '''
    loaded_model = keras.models.load_model('./data/models/mymodel.keras', NeuralNet_MODELS_MAPPER)
        np.testing.assert_allclose(
        NN_model.predict(X_test),
        loaded_model.predict(X_test)
    )
    %tensorboard --logdir ./data/logs  ###################
    %tensorboard --logdir ./data/logs/gradient_tape
    ###--logdir logs/fit
    tensorboard --logdir='Log_Dir'
    '''

    '''
    from tensorboard import program

    tracking_address = logs_path  # the path of your log file.
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    
    import tensorflow as tf
    from tensorboard import main as tb
    tf.flags.FLAGS.logdir = "/path/to/graphs/"
    tb.main()
    '''