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


if __name__ == '__main__':
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    parameters = params_all['neural_net']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    baseline_model_path = Path(args.baseline_model)
    logs_path = Path(args.logs_dir)
    if logs_path.exists():
        shutil.rmtree(logs_path)
    logs_path.mkdir(parents=True)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_keras_path = output_dir / (args.model_name + '.keras')

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(
        parameters[args.model_name]['BUFFER_SIZE']).batch(parameters[args.model_name]['BATCH_SIZE'])

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(parameters[args.model_name]['BATCH_SIZE'])

    # Create an instance of the model
    NN_model = NeuralNet_MODELS_MAPPER.get(args.model_name)(parameters[args.model_name]['NEURONS_CNT'])
    NN_model.build(input_shape=(None, 8))

    loss_object = tf.keras.losses.MeanSquaredError()# Определение функции потерь
    optimizer = tf.keras.optimizers.SGD(learning_rate=parameters[args.model_name]['LEARNING_RATE'])# Определение оптимизатора

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
    train_accuracy = tf.keras.metrics.R2Score(class_aggregation='uniform_average', num_regressors=0, name='train_r2_score', dtype=None)

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_mae = tf.keras.metrics.MeanAbsoluteError(name='test_mae')
    test_accuracy = tf.keras.metrics.R2Score(class_aggregation='uniform_average', num_regressors=0, name='test_r2_score', dtype=None)


    @tf.function
    def train_step(input_vector, labels):     # Обучение одной эпохи
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = NN_model(input_vector, training=True)#получение предсказания
            loss = loss_object(labels, predictions)#вычисление функции потерь
        gradients = tape.gradient(loss, NN_model.trainable_variables)#вычисление градиента
        optimizer.apply_gradients(zip(gradients, NN_model.trainable_variables))# Обновление переменных

        train_loss(loss)
        train_mae(labels, predictions)
        train_accuracy(labels, predictions)


    @tf.function
    def test_step(input_vector, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = NN_model(input_vector, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_mae(labels, predictions)
        test_accuracy(labels, predictions)


    #################################################################################
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = logs_path / 'gradient_tape' / current_time / 'train'
    train_log_dir.mkdir(exist_ok=True, parents=True)
    test_log_dir = logs_path / 'gradient_tape' / current_time / 'test'
    test_log_dir.mkdir(exist_ok=True, parents=True)
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
    test_summary_writer = tf.summary.create_file_writer(str(test_log_dir))

    logdir = logs_path / "fit" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir.mkdir(exist_ok=True, parents=True)
    fit_summary_writer = tf.summary.create_file_writer(str(logdir))

    tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=str(logdir))

    # Процесс обучения
    for epoch in range(parameters[args.model_name]['EPOCHS']):
        # Reset the metrics at the start of the next epoch
        for (x_train, y_train) in train_ds:
            with fit_summary_writer.as_default():
                train_step(x_train, y_train)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('mae', train_mae.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        for (x_test, y_test) in test_ds:
            test_step(x_test, y_test)

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('mae', test_mae.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
        '''
        template = 'Epoch {}, Loss: {}, MAE: {}, Test Loss: {}, Test MAE: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_mae.result(),
                              test_loss.result(),
                              test_mae.result()))
        '''
        template = 'Epoch {}, Loss: {}, MAE: {}, Accuracy: {}, Test Loss: {}, Test MAE: {}, Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_mae.result(),
                              train_accuracy.result(),
                              test_loss.result(),
                              test_mae.result(),
                              test_accuracy.result()))


        # Reset metrics every epoch
        train_loss.reset_state()
        test_loss.reset_state()
        train_accuracy.reset_state()
        test_accuracy.reset_state()

    with fit_summary_writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace", ####################
            step=0,
            profiler_outdir=str(logdir) ###################
        )

    NN_model.save(output_model_keras_path)#

    #Проверка загрузки
    loaded_model = keras.models.load_model(output_model_keras_path, NeuralNet_MODELS_MAPPER)
    print(np.testing.assert_allclose(NN_model.predict(X_test),loaded_model.predict(X_test)))


