import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class Visualization:
    """Класс для визуализации результатов аппроксимации"""

    def __init__(self, data_x, data_y, segments, approx_models):
        self.data_x = data_x
        self.data_y = data_y
        self.segments = segments
        self.approx_models = approx_models
        self.colors = plt.cm.rainbow(np.linspace(0, 1, len(approx_models)))
        self.all_errors = []
        self.all_times = []

    def plot_approximation(self):
        """График аппроксимации сегментов"""
        plt.figure(figsize=(18, 14))
        ax1 = plt.subplot(2, 1, 1)
        plt.scatter(self.data_x, self.data_y, alpha=0.3, label='Исходные данные', color='lightblue')

        for i, (seg, model_info) in enumerate(zip(self.segments, self.approx_models)):
            data_x_seg = seg['time']
            data_y_seg = seg['velocity']

            model = model_info['model']
            scaler_X = model_info['scaler_X']
            scaler_y = model_info['scaler_y']
            params = model_info['params']

            # Генерация точек для гладкой кривой
            x_fit = np.linspace(data_x_seg.min(), data_x_seg.max(), 100)
            x_fit_scaled = scaler_X.transform(x_fit.reshape(-1, 1))

            # Получаем параметры модели
            k, a, b, c, d = params['k'], params['a'], params['b'], params['c'], params['d']
            y_fit = k * x_fit + a + b * np.sin(c * x_fit + d)

            # Преобразуем входные данные и делаем предсказание
            x_seg_scaled = scaler_X.transform(data_x_seg.reshape(-1, 1))
            predicted_params = model.predict(x_seg_scaled)
            k_pred, a_pred, b_pred, c_pred, d_pred = predicted_params.T

            y_pred_scaled = k_pred * x_seg_scaled[:, 0] + a_pred + b_pred * np.sin(c_pred * x_seg_scaled[:, 0] + d_pred)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

            # Вычисляем ошибки
            errors = np.abs(y_pred.flatten() - data_y_seg.flatten())
            r2 = r2_score(data_y_seg, y_pred)

            # Сохраняем ошибки для графика
            self.all_errors.extend(errors.tolist())
            self.all_times.extend(data_x_seg.tolist())

            # Отрисовка аппроксимационной кривой
            plt.plot(x_fit, y_fit, linestyle='-', linewidth=2, color=self.colors[i],
                     label=f'Сегмент {i + 1}: y = {k:.2f}x + {a:.2f} + {b:.2f}sin({c:.2f}x + {d:.2f})')

            plt.text(data_x_seg.mean(), y_fit.min(), f'R²={r2:.3f}',
                     ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))

        plt.xlabel('Время (с)')
        plt.ylabel('Перемещение (мм)')
        plt.title('Аппроксимация сегментов моделями')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()

    def plot_errors(self):
        """График абсолютной погрешности"""
        plt.subplot(2, 1, 2)
        plt.plot(self.all_times, self.all_errors, 'b-', alpha=0.5, label='Абсолютная погрешность')
        plt.fill_between(self.all_times, 0, self.all_errors, color='blue', alpha=0.1)

        plt.title("Абсолютная погрешность аппроксимации", pad=20)
        plt.xlabel("Время, с", labelpad=10)
        plt.ylabel("Погрешность, мм", labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def visualize_results(self):
        """Выполняет все визуализации"""
        self.plot_approximation()
        self.plot_errors()
