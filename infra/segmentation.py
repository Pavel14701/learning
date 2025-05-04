from typing import List, Tuple

import keras._tf_keras.keras as keras  # type: ignore
import numpy as np
from keras._tf_keras.keras.layers import (  # type: ignore
    BatchNormalization,
    Dense,
    Dropout,
)
from keras._tf_keras.keras.optimizers import Adam  # type: ignore
from scipy.signal import savgol_filter  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore


class SegmentationModel:
    """Класс для обучения и предсказания сегментов"""

    def __init__(self) -> None:
        """Инициализация модели"""
        self.scaler: MinMaxScaler = MinMaxScaler()
        self.model: keras.Sequential = self._build_model()

    def _build_model(self) -> keras.Sequential:
        """Создание модели сегментации"""
        model = keras.Sequential([
            Dense(64, activation='relu', input_dim=13), 
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=[
                'accuracy', 
                keras.metrics.Precision(), 
                keras.metrics.Recall()]
        )
        return model

    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[keras.Sequential, MinMaxScaler]:
        """Обучение модели"""
        X_scaled: np.ndarray = self.scaler.fit_transform(X)
        self.model.fit(
            X_scaled, y, epochs=100, batch_size=16, 
            validation_split=0.2, verbose=1,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=10
                )
            ]
        )
        return self.model, self.scaler

    def predict_segments(
        self, model: keras.Sequential, scaler: MinMaxScaler,
        time: np.ndarray, data: np.ndarray,
        window_size: int = 15, min_segment_length: int = 25, max_segments: int = 40
    ) -> List[Tuple[int, int]]:
        """Метод для предсказания сегментов"""

        smoothed_data: np.ndarray = savgol_filter(data, 21, 3)
        first_deriv: np.ndarray = np.gradient(smoothed_data)
        second_deriv: np.ndarray = np.gradient(first_deriv)

        trend_changes: np.ndarray = np.flatnonzero(np.diff(np.sign(first_deriv)))
        indices: np.ndarray = trend_changes[
            (trend_changes >= window_size) & (trend_changes < len(time) - window_size)
        ]

        left_indices: np.ndarray = np.add.outer(indices, np.arange(-window_size, 0)).T
        right_indices: np.ndarray = np.add.outer(indices, np.arange(0, window_size)).T

        # 🔹 Исправлено: Приводим `left_trend` и `right_trend` к `shape (indices.shape,)`
        left_trend: np.ndarray = np.full(indices.shape, np.polyfit(
            time[left_indices].flatten(), smoothed_data[left_indices].flatten(), 1
        )[0])
        
        right_trend: np.ndarray = np.full(indices.shape, np.polyfit(
            time[right_indices].flatten(), smoothed_data[right_indices].flatten(), 1
        )[0])

        trend_ratio: np.ndarray = np.divide(
            right_trend, left_trend, out=np.zeros_like(right_trend), where=left_trend != 0
        )
        sign_change: np.ndarray = np.full(indices.shape, np.sign(left_trend) != np.sign(right_trend))
        curvature: np.ndarray = second_deriv[indices] * np.sign(right_trend - left_trend)

        # 🔹 Приведение `mean()` и `std()` к `shape (indices.shape,)`
        mean_left_first = np.mean(first_deriv[left_indices], axis=0).reshape(indices.shape)
        mean_right_first = np.mean(first_deriv[right_indices], axis=0).reshape(indices.shape)

        std_left_first = np.std(first_deriv[left_indices], axis=0).reshape(indices.shape)
        std_right_first = np.std(first_deriv[right_indices], axis=0).reshape(indices.shape)

        mean_left_second = np.mean(second_deriv[left_indices], axis=0).reshape(indices.shape)
        mean_right_second = np.mean(second_deriv[right_indices], axis=0).reshape(indices.shape)

        # 🔹 Убеждаемся, что все массивы имеют одинаковый shape
        all_arrays = [
            left_trend, right_trend, trend_ratio, curvature,
            mean_left_first, mean_right_first,
            std_left_first, std_right_first,
            sign_change, np.abs(left_trend - right_trend),
            second_deriv[indices], mean_left_second, mean_right_second
        ]

        # 🔹 Проверяем размер каждого массива перед column_stack()
        for i, arr in enumerate(all_arrays):
            print(f"Массив {i}: shape {arr.shape}")

        features: np.ndarray = np.column_stack(all_arrays)

        X_scaled: np.ndarray = scaler.transform(features)
        probs: np.ndarray = model.predict(X_scaled).flatten()

        extra_points: np.ndarray = indices[probs > 0.5]
        main_points: np.ndarray = np.unique(
            np.concatenate([
                trend_changes, extra_points,
                np.flatnonzero(np.abs(second_deriv) > 1.5 * np.std(second_deriv))
            ])
        )

        main_points = main_points[
            (main_points >= min_segment_length) & (main_points < len(time) - min_segment_length)
        ]

        final_points: np.ndarray = np.array([
            p for i, p in enumerate(main_points) if i == 0 or (p - main_points[i - 1] > min_segment_length // 2)
        ])

        segments: List[Tuple[int, int]] = [
            (start, end) for start, end in zip([0] + final_points.tolist(), final_points.tolist() + [len(time) - 1])
        ]

        return segments[:max_segments]
