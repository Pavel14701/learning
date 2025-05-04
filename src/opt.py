import numpy as np
import tensorflow as tf
import asyncio
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
import keras._tf_keras.keras as keras
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.layers import Dense, BatchNormalization, Dropout






class ApproximationModel:
    """Класс для обучения модели аппроксимации"""

    def __init__(self, learning_rate=0.00001):
        self.model = self._build_model()
        self.learning_rate = learning_rate

    def _build_model(self):
        """Создание модели для аппроксимации kx + a + b*sin(cx + d)"""
        model = keras.Sequential([
            Dense(128, activation='relu', input_shape=(1,)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(5)
        ])
        return model

    @staticmethod
    def custom_loss(y_true, y_pred):
        """Кастомная функция потерь для модели аппроксимации"""
        k, a, b, c, d = tf.split(y_pred, num_or_size_splits=5, axis=1)
        x, y_true_values = tf.split(y_true, num_or_size_splits=2, axis=1)
        y_pred_values = k * x + a + b * tf.sin(c * x + d)
        mse_loss = keras.losses.Huber()(y_true_values, y_pred_values)
        reg_loss = 0.001 * tf.reduce_mean(tf.square(k) + tf.square(a) + tf.square(b) + tf.square(c) + tf.square(d))
        return mse_loss + reg_loss

    async def train_segment_async(self, seg):
        """Асинхронное обучение одного сегмента"""
        time_seg = seg['time'].reshape(-1, 1)
        vel_seg = seg['velocity'].reshape(-1, 1)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(time_seg)
        y_scaled = scaler_y.fit_transform(vel_seg)
        X_scaled_tf = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
        y_true_for_loss_tf = tf.convert_to_tensor(np.column_stack([X_scaled, y_scaled]), dtype=tf.float32)

        with tf.device('/GPU:0'):
            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(optimizer=optimizer, loss=self.custom_loss)
            await asyncio.to_thread(self.model.fit, X_scaled_tf, y_true_for_loss_tf, epochs=300, verbose=1)
        return self.model

    async def train_parallel_async(self, segments):
        """Запускаем обучение всех сегментов параллельно"""
        tasks = [self.train_segment_async(seg) for seg in segments]
        models = await asyncio.gather(*tasks)
        return models
