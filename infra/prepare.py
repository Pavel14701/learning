from typing import Tuple
import numpy as np
from scipy.ndimage import uniform_filter1d  # type: ignore

class DataProcessor:
    """
    Класс для обработки данных 
    (сегментация, нормализация и вычисления)
    """

    @staticmethod
    def prepare_segmentation_data(
        time: np.ndarray, 
        data: np.ndarray, 
        window_size: int = 9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для модели сегментации
        
        Аргументы:
        - time: массив временных меток (np.ndarray).
        - data: массив значений (np.ndarray).
        - window_size: размер окна сегментации (int, по умолчанию = 9).
        
        Возвращает:
        - features: признаки сегментации (np.ndarray).
        - y: метки классов (np.ndarray).
        """

        # 🔹 Ограничение window_size, чтобы избежать выхода за границы
        window_size = min(window_size, len(time) // 2)

        kernel_size: int = 15
        smoothed: np.ndarray = uniform_filter1d(data, size=kernel_size)

        first_deriv: np.ndarray = np.gradient(smoothed)
        second_deriv: np.ndarray = np.gradient(first_deriv)

        first_deriv = uniform_filter1d(first_deriv, size=5)
        second_deriv = uniform_filter1d(second_deriv, size=5)

        indices: np.ndarray = np.arange(window_size, len(time) - window_size)
        left_indices: np.ndarray = np.add.outer(
            indices, 
            np.arange(-window_size, 0)
        ).T
        right_indices: np.ndarray = np.add.outer(
            indices, 
            np.arange(0, window_size)
        ).T

        # 🔹 Исправлено: Приводим к 1D формату перед передачей в polyfit()
        left_trend: np.ndarray = np.polyfit(
            time[left_indices].reshape(-1), 
            smoothed[left_indices].reshape(-1), 
            1
        )[0]
        right_trend: np.ndarray = np.polyfit(
            time[right_indices].reshape(-1), 
            smoothed[right_indices].reshape(-1), 
            1
        )[0]

        # 🔹 Проверяем размеры массивов перед column_stack()
        print(f"Размеры массивов перед column_stack:")
        print(f"left_trend.shape: {left_trend.shape}, right_trend.shape: {right_trend.shape}, indices.shape: {indices.shape}")

        trend_ratio: np.ndarray = np.divide(
            right_trend, left_trend, 
            out=np.zeros_like(right_trend), 
            where=left_trend != 0
        )
        sign_change: np.ndarray = np.sign(left_trend) != np.sign(right_trend)
        curvature: np.ndarray = second_deriv[indices] * np.sign(right_trend - left_trend)

        # 🔹 Убеждаемся, что все массивы имеют одинаковый shape
        all_arrays = [
            left_trend, right_trend, trend_ratio, curvature,
            np.mean(first_deriv[left_indices], axis=1),
            np.mean(first_deriv[right_indices], axis=1),
            np.std(first_deriv[left_indices], axis=1),
            np.std(first_deriv[right_indices], axis=1),
            sign_change, np.abs(left_trend - right_trend),
            second_deriv[indices],
            np.mean(second_deriv[left_indices], axis=1),
            np.mean(second_deriv[right_indices], axis=1)
        ]

        # 🔹 Проверяем размер каждого массива перед column_stack()
        for i, arr in enumerate(all_arrays):
            print(f"Массив {i}: shape {arr.shape}")

        features: np.ndarray = np.column_stack(all_arrays)

        stable_sign_change: np.ndarray = (
            sign_change
        ) & (
            np.abs(left_trend) > 0.1 * np.max(np.abs(first_deriv))
        ) & (
            np.abs(right_trend) > 0.1 * np.max(np.abs(first_deriv))
        )
        strong_curvature: np.ndarray = np.abs(
            second_deriv[indices]
        ) > 2 * np.std(np.abs(second_deriv))

        y: np.ndarray = np.where(stable_sign_change | strong_curvature, 1, 0)
        return features, y
