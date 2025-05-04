import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import keras._tf_keras.keras as keras
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.layers import Dense, BatchNormalization, Dropout


class SegmentationModel:
    """Класс для обучения и предсказания сегментов"""

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = self._build_model()

    def _build_model(self):
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
        model.compile(optimizer=Adam(learning_rate=0.0005),
                      loss='binary_crossentropy',
                      metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
        return model

    def train(self, X, y):
        """Обучение модели"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, epochs=100, batch_size=16, validation_split=0.2, verbose=1,
                       callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)])
        return self.model, self.scaler

    def predict_segments(self, model, scaler, time, data, window_size=15, min_segment_length=25, max_segments=40):
        """Метод для предсказания сегментов"""
        smoothed_data = savgol_filter(data, 21, 3)
        first_deriv = np.gradient(smoothed_data)
        second_deriv = np.gradient(first_deriv)
        trend_changes = np.flatnonzero(np.diff(np.sign(first_deriv)))

        indices = trend_changes[(trend_changes >= window_size) & (trend_changes < len(time) - window_size)]
        left_indices = np.add.outer(indices, np.arange(-window_size, 0))
        right_indices = np.add.outer(indices, np.arange(0, window_size))

        left_trend = np.polyfit(time[left_indices.T], smoothed_data[left_indices.T], deg=1)[0]
        right_trend = np.polyfit(time[right_indices.T], smoothed_data[right_indices.T], deg=1)[0]

        trend_ratio = np.divide(right_trend, left_trend, out=np.zeros_like(right_trend), where=left_trend!=0)
        sign_change = np.sign(left_trend) != np.sign(right_trend)
        curvature = second_deriv[indices] * np.sign(right_trend - left_trend)

        features = np.column_stack([
            left_trend, right_trend, trend_ratio, curvature,
            np.mean(first_deriv[left_indices.T], axis=1), np.mean(first_deriv[right_indices.T], axis=1),
            np.std(first_deriv[left_indices.T], axis=1), np.std(first_deriv[right_indices.T], axis=1),
            sign_change, np.abs(left_trend - right_trend), second_deriv[indices],
            np.mean(second_deriv[left_indices.T], axis=1), np.mean(second_deriv[right_indices.T], axis=1)
        ])

        X_scaled = scaler.transform(features)
        probs = model.predict(X_scaled).flatten()
        extra_points = indices[probs > 0.5]

        main_points = np.unique(np.concatenate([trend_changes, extra_points, np.flatnonzero(np.abs(second_deriv) > 1.5 * np.std(second_deriv))]))
        main_points = main_points[(main_points >= min_segment_length) & (main_points < len(time) - min_segment_length)]

        final_points = np.array([p for i, p in enumerate(main_points) if i == 0 or (p - main_points[i - 1] > min_segment_length // 2)])
        segments = [(start, end) for start, end in zip([0] + final_points.tolist(), final_points.tolist() + [len(time) - 1])]
        return segments[:max_segments]
