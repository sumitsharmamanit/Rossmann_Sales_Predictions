import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class DeepNeuralNetwork:
    def __init__(self):
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.adam = Adam(lr=0.001, decay=1e-3)
        self.model = None

    def generate_timeseries(self, x_train, y_train, x_val, y_val, x_test, n_input, sr, strid, bz):
        self.train_generator = TimeseriesGenerator(x_train, y_train, length=n_input, sampling_rate=sr, stride=strid,
                                                   batch_size=bz)
        self.val_generator = TimeseriesGenerator(x_val, y_val, length=n_input, sampling_rate=sr, stride=strid,
                                                 batch_size=bz)
        self.test_generator = TimeseriesGenerator(x_test, np.zeros((x_test.shape[0], 1)), length=n_input,
                                                  sampling_rate=sr,
                                                  stride=strid, batch_size=bz)

    def create_model(self, n_input, n_feature):
        self.model = Sequential()
        self.model.add(LSTM(128, activation='relu', input_shape=(n_input, n_feature), return_sequences=True))
        self.model.add(Dropout(0.25))
        self.model.add(LSTM(64, activation='relu', return_sequences=False))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(1))
        self.model.compile(loss="mean_squared_error", optimizer=self.adam,
                           metrics=[MeanSquaredError(), RootMeanSquaredError()])
        self.model.summary()

    def fit_model(self, epoch):
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min', min_delta=0.0001)
        history = self.model.fit(self.train_generator, validation_data=self.val_generator, epochs=epoch, callbacks=[early_stopping, rlr])
        return history

    def save_model(self, path):
        self.model.save(path+'/LSTM_NN19_v1')

    def load_model(self, path):
        model2 = tf.keras.models.load_model(path+'/LSTM_NN19_v1')
        model2.summary()
        return model2

    def predict(self, loaded_model):
        predict = loaded_model.predict(self.test_generator)
        return predict
