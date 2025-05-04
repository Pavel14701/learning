import asyncio
import time
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from infra.approximation import ApproximationModel
from infra.prepare import DataProcessor
from infra.segmentation import SegmentationModel
from infra.visual import Visualization


async def start_async(dataframe_x: pd.Series, dataframe_y: pd.Series) -> List[Dict[str, Any]]:
    """Асинхронная сегментация и подготовка данных"""
    
    data_x: np.ndarray = np.asarray(dataframe_x.to_numpy(), dtype=np.float64)
    data_y: np.ndarray = np.asarray(dataframe_y.to_numpy(), dtype=np.float64)

    X, y = DataProcessor.prepare_segmentation_data(data_x, data_y)

    segmentation_model = SegmentationModel()
    await asyncio.to_thread(segmentation_model.train_async, X, y)  # 🔹 Асинхронное обучение модели

    segments_indices = segmentation_model.predict_segments(data_x, data_y)

    segments = [{
        "time": np.asarray(data_x[start:end + 1], dtype=np.float64),
        "velocity": np.asarray(data_y[start:end + 1], dtype=np.float64),
        "start_idx": start,
        "end_idx": end
    } for start, end in segments_indices]

    return data_x, data_y, segments


async def train_and_visualize_async(data_x: np.ndarray, data_y: np.ndarray, segments: List[Dict[str, Any]]) -> None:
    """Асинхронное обучение и визуализация"""
    
    approximation_model = ApproximationModel()
    
    # 🔹 **Запускаем асинхронное обучение в 100 потоках**
    approx_models = await approximation_model.train_parallel_async(segments)

    visualizer = Visualization(data_x, data_y, segments, approx_models)
    visualizer.visualize_results()


# 🚀 **Запуск кода**
async def main():
    df = pd.read_excel("dataset.xlsx", skiprows=2)
    data_x = df["Время, с"]
    data_y = df["Скорость v, м/с"]
    
    start_time = time.time()

    # 🔹 **Асинхронная сегментация**
    data_x, data_y, segments = await start_async(data_x, data_y)

    # 🔹 **Асинхронное обучение и визуализация**
    await train_and_visualize_async(data_x, data_y, segments)

    end_time = time.time()
    print(f"✅ Время выполнения: {end_time - start_time:.2f} секунд")


if __name__ == "__main__":
    asyncio.run(main())
