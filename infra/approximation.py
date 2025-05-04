import asyncio
from typing import Any, Dict, List

import keras._tf_keras.keras as keras  # type: ignore
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.layers import (  # type: ignore
    BatchNormalization,
    Dense,
    Dropout,
)
from keras._tf_keras.keras.optimizers import Adam  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore 


class ApproximationModel:
    """Класс для обучения модели аппроксимации"""

    def __init__(self, learning_rate: float = 0.00001) -> None:
        """Инициализация модели"""
        self.model: keras.Sequential = self._build_model()
        self.learning_rate: float = learning_rate

        # 🔹 Установка доступных GPU перед началом асинхронного обучения
        self._set_gpu_device()

    def _set_gpu_device(self) -> None:
        """Гарантированное использование GPU"""
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            tf.config.set_visible_devices(gpu_devices[0], 'GPU')
            print("🚀 Используется GPU:", gpu_devices[0])
        else:
            print("⚠ GPU не обнаружена, используется CPU")

    def _build_model(self) -> keras.Sequential:
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
    def custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Кастомная функция потерь для модели аппроксимации"""
        k, a, b, c, d = tf.split(y_pred, num_or_size_splits=5, axis=1)
        x, y_true_values = tf.split(y_true, num_or_size_splits=2, axis=1)
        y_pred_values = k * x + a + b * tf.sin(c * x + d)
        mse_loss = keras.losses.Huber()(y_true_values, y_pred_values)
        reg_loss = 0.001 * tf.reduce_mean(
            tf.square(k) + tf.square(a) + tf.square(b) + tf.square(c) + tf.square(d)
        )
        return mse_loss + reg_loss

    async def train_segment_async(
        self, 
        seg: Dict[str, np.ndarray]
    ) -> keras.Sequential:
        """Асинхронное обучение одного сегмента"""
        time_seg: np.ndarray = seg['time'].reshape(-1, 1)
        vel_seg: np.ndarray = seg['velocity'].reshape(-1, 1)
        scaler_X: StandardScaler = StandardScaler()
        scaler_y: StandardScaler = StandardScaler()
        X_scaled: np.ndarray = scaler_X.fit_transform(time_seg)
        y_scaled: np.ndarray = scaler_y.fit_transform(vel_seg)
        X_scaled_tf: tf.Tensor = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
        y_true_for_loss_tf: tf.Tensor = tf.convert_to_tensor(
            np.column_stack([X_scaled, y_scaled]), 
            dtype=tf.float32
        )

        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.custom_loss)

        # 🔹 Удалён `tf.device('/GPU:0')`, теперь GPU выбирается заранее
        await asyncio.to_thread(
            self.model.fit, 
            X_scaled_tf, 
            y_true_for_loss_tf, 
            epochs=300, 
            verbose=1
        )
        return self.model

    async def train_parallel_async(
        self, 
        segments: List[Dict[str, Any]]
    ) -> List[keras.Sequential]:
        """Запускаем обучение всех сегментов параллельно"""
        tasks = [self.train_segment_async(seg) for seg in segments]
        models = await asyncio.gather(*tasks)
        return models
