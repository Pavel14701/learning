import numpy as np
from scipy.ndimage import uniform_filter1d


class DataProcessor:
    """Класс для обработки данных (сегментация, нормализация и вычисления)"""

    @staticmethod
    def prepare_segmentation_data(time, data, window_size=9):
        """Подготовка данных для модели сегментации"""
        kernel_size = 15
        smoothed = uniform_filter1d(data, size=kernel_size)

        first_deriv = np.gradient(smoothed)
        second_deriv = np.gradient(first_deriv)

        first_deriv = uniform_filter1d(first_deriv, size=5)
        second_deriv = uniform_filter1d(second_deriv, size=5)

        indices = np.arange(window_size, len(time) - window_size)
        left_indices = np.add.outer(indices, np.arange(-window_size, 0))
        right_indices = np.add.outer(indices, np.arange(0, window_size))

        left_trend = np.polyfit(time[left_indices.T], smoothed[left_indices.T], 1)[0]
        right_trend = np.polyfit(time[right_indices.T], smoothed[right_indices.T], 1)[0]

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

        stable_sign_change = (sign_change) & (np.abs(left_trend) > 0.1 * np.max(np.abs(first_deriv))) & \
                             (np.abs(right_trend) > 0.1 * np.max(np.abs(first_deriv)))
        strong_curvature = np.abs(second_deriv[indices]) > 2 * np.std(np.abs(second_deriv))

        y = np.where(stable_sign_change | strong_curvature, 1, 0)
        return features, y