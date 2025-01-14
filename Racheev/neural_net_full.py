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
    def __init__(self, neurons_cnt=128, **kwargs):
        super(NeuralNet, self).__init__(**kwargs)
        self.neurons_cnt = neurons_cnt  # Сохраняем значение параметра для конфигурации
        self.d_in = Dense(8, activation='relu')
        self.d1 = Dense(neurons_cnt, activation='relu')
        self.d2 = Dense(neurons_cnt, activation='relu')
        self.d_out = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.d_in(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d_out(x)

    def get_config(self):
        # Возвращаем параметры модели, включая кастомные
        config = super(NeuralNet, self).get_config()
        config.update({
            "neurons_cnt": self.neurons_cnt  # Добавляем кастомный параметр в конфигурацию
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
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib',
                        required=False, help='path to linear regression prod version')
    parser.add_argument('--logs_dir', '-lg', type=str, default='data/logs',
                        required=False, help='path to save logs folder')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()