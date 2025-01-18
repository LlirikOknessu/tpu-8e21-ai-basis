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
    def __init__(self, neurons_cnt_input = 8, neurons_cnt_d1 = 128, neurons_cnt_d2 = 128, **kwargs):
        super(NeuralNet, self).__init__(**kwargs)
        self.neurons_cnt_input = neurons_cnt_input  # Сохраняем значение параметра для конфигурации
        self.neurons_cnt_d1 = neurons_cnt_d1
        self.neurons_cnt_d2 = neurons_cnt_d2
        self.d_in = Dense(neurons_cnt_input, activation='relu')
        self.d1 = Dense(neurons_cnt_d1, activation='relu')
        self.d2 = Dense(neurons_cnt_d2, activation='relu')
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
            "neurons_cnt_input": self.neurons_cnt_input,  #  Добавляем кастомный параметр в конфигурацию
            "neurons_cnt_d1": self.neurons_cnt_d1,
            "neurons_cnt_d2": self.neurons_cnt_d2
        })

        return config

    @classmethod
    def from_config(cls, config):
        # Создаём экземпляр класса из конфигурации
        return cls(**config)


NeuralNet_MODELS_MAPPER = {'NeuralNet': NeuralNet}

NeuralNet_MODELS_BEST_PARAMETERS = {
    'NeuralNet': {'INPUT_DENSE' : 12, 'NEURONS_CNT_D1' : 64, 'NEURONS_CNT_D2' : 64,
        'BATCH_SIZE' : 128, 'BUFFER_SIZE' : 256, 'LEARNING_RATE' : 0.002, 'EPOCHS' : 5000}}

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


if __name__ == '__main__':
    random.seed(67)
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    parameters = params_all['neural_net']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    logs_path = Path(args.logs_dir)
    logs_path.mkdir(exist_ok=True, parents=True)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_keras_path = output_dir / (args.model_name + '_prod.keras')

    best_params = NeuralNet_MODELS_BEST_PARAMETERS.get(args.model_name)

    X_train_name = input_dir / 'X_full.csv'
    y_train_name = input_dir / 'y_full.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    y_train_cols = y_train.columns

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(
        best_params['BUFFER_SIZE']).batch(best_params['BATCH_SIZE'])

    # Create an instance of the model
    NN_model = NeuralNet_MODELS_MAPPER.get(args.model_name)(parameters[args.model_name]['INPUT_DENSE'],
                                                            parameters[args.model_name]['NEURONS_CNT_D1'],
                                                            parameters[args.model_name]['NEURONS_CNT_D2'])
    NN_model.build(input_shape=(None, parameters[args.model_name]['INPUT_DENSE']))

    loss_object = tf.keras.losses.MeanSquaredError()  # Определение функции потерь
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=best_params['LEARNING_RATE'])  # Определение оптимизатора

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
    train_accuracy = tf.keras.metrics.R2Score(class_aggregation='uniform_average', num_regressors=0,
                                              name='train_r2_score', dtype=None)

    @tf.function
    def train_step(input_vector, labels):  # Обучение одной эпохи
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = NN_model(input_vector, training=True)  # получение предсказания
            loss = loss_object(labels, predictions)  # вычисление функции потерь
        gradients = tape.gradient(loss, NN_model.trainable_variables)  # вычисление градиента
        optimizer.apply_gradients(zip(gradients, NN_model.trainable_variables))  # Обновление переменных

        train_loss(loss)
        train_mae(labels, predictions)
        train_accuracy(labels, predictions)


    #################################################################################
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = logs_path / 'gradient_tape' / current_time / 'full'
    train_log_dir.mkdir(exist_ok=True, parents=True)
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))

    logdir = logs_path / "fit" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir.mkdir(exist_ok=True, parents=True)
    fit_summary_writer = tf.summary.create_file_writer(str(logdir))

    tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=str(logdir))

    def log_weights(epoch):
        for layer in NN_model.layers:
            weights, biases = layer.get_weights()  # Получаем веса и смещения слоя
            tf.summary.histogram(f'weights/{layer.name}', weights, step=epoch)  # Логируем веса
            tf.summary.histogram(f'biases/{layer.name}', biases, step=epoch)  # Логируем смещения


    # Процесс обучения
    for epoch in range(best_params['EPOCHS']):
        # Reset the metrics at the start of the next epoch
        for (x_train, y_train) in train_ds:
            with fit_summary_writer.as_default():
                train_step(x_train, y_train)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('mae', train_mae.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            log_weights(epoch)

        '''
        template = 'Epoch {}, Loss: {}, MAE: {}, Test Loss: {}, Test MAE: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_mae.result(),
                              test_loss.result(),
                              test_mae.result()))
        '''
        # Reset metrics every epoch
        train_loss.reset_state()
        train_accuracy.reset_state()

    template = 'Epoch {}, Loss: {}, MAE: {}, Accuracy: {}'
    print(template.format(best_params['EPOCHS']+1,
                          train_loss.result(),
                          train_mae.result(),
                          train_accuracy.result()))



    with fit_summary_writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace_prod",  ####################
            step=0,
            profiler_outdir=str(logdir)  ###################
        )

    NN_model.save(output_model_keras_path)