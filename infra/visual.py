from operator import itemgetter
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from keras._tf_keras.keras.models import Sequential  # type: ignore
from sklearn.metrics import r2_score  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore


class Visualization:
    """Класс для визуализации результатов аппроксимации"""
    def __init__(
        self, 
        data_x: np.ndarray, 
        data_y: np.ndarray, 
        segments: List[Dict[str, Any]], 
        approx_models: List[Dict[str, Any]]
    ) -> None:
        """Инициализация визуализации"""
        self.data_x: np.ndarray = data_x
        self.data_y: np.ndarray = data_y
        self.segments: List[Dict[str, np.ndarray]] = segments
        self.approx_models: List[Dict[str, Any]] = approx_models
        self.colors: np.ndarray = np.array(
            plt.get_cmap('rainbow')(
                np.linspace(0, 1, len(approx_models))
            )
        )
        self.all_errors: List[float] = []
        self.all_times: List[float] = []

    def plot_approximation(self) -> None:
        """График аппроксимации сегментов"""
        plt.figure(figsize=(18, 14))
        plt.scatter(
            self.data_x, 
            self.data_y, 
            alpha=0.3, 
            label='Исходные данные', 
            color='lightblue'
        )

        for i, (
            seg, model_info
        ) in enumerate(
            zip(self.segments, self.approx_models)
        ):
            data_x_seg: np.ndarray = seg['time']
            data_y_seg: np.ndarray = seg['velocity']

            model: Sequential = model_info['model']
            scaler_X: MinMaxScaler = model_info['scaler_X']
            scaler_y: MinMaxScaler = model_info['scaler_y']
            params = model_info['params']

            # Генерация точек для гладкой кривой
            x_fit: np.ndarray = np.linspace(
                start=data_x_seg.min(), 
                stop=data_x_seg.max(), 
                num=100
            )

            # Получаем параметры модели
            k, a, b, c, d = itemgetter('k', 'a', 'b', 'c', 'd')(params)

            y_fit: np.ndarray = k * x_fit + a + b * np.sin(c * x_fit + d)

            # Преобразуем входные данные и делаем предсказание
            x_seg_scaled: np.ndarray = scaler_X.transform(data_x_seg.reshape(-1, 1))
            predicted_params: np.ndarray = model.predict(x_seg_scaled)
            k_pred, a_pred, b_pred, c_pred, d_pred = predicted_params.T

            y_pred_scaled: np.ndarray = k_pred * x_seg_scaled[:, 0] + \
                a_pred + b_pred * np.sin(
                    c_pred * x_seg_scaled[:, 0] + d_pred
            )
            y_pred: np.ndarray = scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            )

            # Вычисляем ошибки
            errors: np.ndarray = np.abs(y_pred.flatten() - data_y_seg.flatten())
            r2: float = r2_score(data_y_seg, y_pred)

            # Сохраняем ошибки для графика
            self.all_errors.extend(errors.tolist())
            self.all_times.extend(data_x_seg.tolist())

            # Отрисовка аппроксимационной кривой
            plt.plot(
                x_fit, y_fit, linestyle='-', linewidth=2, color=self.colors[i],
                label=f'Сегмент {i + 1}: y = \
                    {k:2f}x + {a:2f} + {b:2f}\
                    sin({c:2f}x + {d:2f}')

            plt.text(data_x_seg.mean(), y_fit.min(), f'R²={r2:.3f}',
                     ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))

        plt.xlabel('Время (с)')
        plt.ylabel('Перемещение (мм)')
        plt.title('Аппроксимация сегментов моделями')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()

    def plot_errors(self) -> None:
        """График абсолютной погрешности"""
        plt.subplot(2, 1, 2)
        plt.plot(
            self.all_times, 
            self.all_errors, 
            'b-', 
            alpha=0.5, 
            label='Абсолютная погрешность'
        )
        plt.fill_between(
            x=self.all_times, 
            y1=0, 
            y2=self.all_errors, 
            color='blue', 
            alpha=0.1
        )

        plt.title("Абсолютная погрешность аппроксимации", pad=20)
        plt.xlabel("Время, с", labelpad=10)
        plt.ylabel("Погрешность, мм", labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def visualize_results(self) -> None:
        """Выполняет все визуализации"""
        self.plot_approximation()
        self.plot_errors()
