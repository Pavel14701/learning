import asyncio
import time

import numpy as np
import pandas as pd

from infra.approximation import ApproximationModel
from infra.prepare import DataProcessor
from infra.segmentation import SegmentationModel
from infra.visual import Visualization


def start(dataframe_x: pd.Series, dataframe_y: pd.Series) -> None:
    """Запуск сегментации, аппроксимации и визуализации"""

    start_time = time.time()

    # 🔹 **Преобразование данных в np.ndarray**
    data_x: np.ndarray = np.asarray(dataframe_x.to_numpy(), dtype=np.float64)
    data_y: np.ndarray = np.asarray(dataframe_y.to_numpy(), dtype=np.float64)

    # 🔹 **Подготовка данных**
    X, y = DataProcessor.prepare_segmentation_data(data_x, data_y)

    # 🔹 **Обучение модели сегментации**
    segmentation_model = SegmentationModel()
    model, scaler = segmentation_model.train(X, y)

    # 🔹 **Предсказание сегментов**
    segments_indices = segmentation_model.predict_segments(
        model, scaler, data_x, data_y
    )

    # 🔹 **Формирование списка сегментов** (с гарантированным `np.ndarray`)
    segments = [{
        'time': np.asarray(data_x[start:end + 1], dtype=np.float64),
        'velocity': np.asarray(data_y[start:end + 1], dtype=np.float64),
        'start_idx': start,
        'end_idx': end
    } for start, end in segments_indices]

    # 🔹 **Проверка типов**
    for seg in segments:
        print(f"time type: {type(seg['time'])}, velocity type: {type(seg['velocity'])}")

    # 🔹 **Асинхронное обучение моделей аппроксимации**
    approximation_model = ApproximationModel()
    approx_models = asyncio.run(approximation_model.train_parallel_async(segments))

    # 🔹 **Визуализация результатов**
    visualizer = Visualization(data_x, data_y, segments, approx_models)
    visualizer.visualize_results()

    # 🔹 **Вывод времени выполнения**
    end_time = time.time()
    print(f"✅ Время выполнения: {end_time - start_time:.2f} секунд")


if __name__ == "__main__":
    df = pd.read_excel(
        "dataset.xlsx",
        skiprows=2)
    data_x = df["Время, с"]
    data_y = df["Скорость v, м/с"]
    start(data_x, data_y)