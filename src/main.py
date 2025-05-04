import time
import asyncio
import pandas as pd

# Импорт классов из модуля `src.infra`
from src.infra.approximation import ApproximationModel
from src.infra.prepare import DataProcessor
from src.infra.segmentation import SegmentationModel
from src.infra.visual import Visualization

# Засекаем время начала выполнения
start_time = time.time()

# 🔹 **Загрузка данных**
dataframe = pd.read_excel("/content/drive/MyDrive/dataset.xlsx", skiprows=2)
data_x = dataframe["Время, с"].values
data_y = dataframe["Скорость v, м/с"].values

# 🔹 **Подготовка данных**
X, y = DataProcessor.prepare_segmentation_data(data_x, data_y)

# 🔹 **Обучение модели сегментации**
segmentation_model = SegmentationModel()
model, scaler = segmentation_model.train(X, y)

# 🔹 **Предсказание сегментов**
segments_indices = segmentation_model.predict_segments(model, scaler, data_x, data_y)

# 🔹 **Формирование списка сегментов**
segments = [{'time': data_x[start:end + 1], 'velocity': data_y[start:end + 1], 'start_idx': start, 'end_idx': end}
            for start, end in segments_indices]

# 🔹 **Асинхронное обучение моделей аппроксимации**
approximation_model = ApproximationModel()
approx_models = asyncio.run(approximation_model.train_parallel_async(segments))

# 🔹 **Визуализация результатов**
visualizer = Visualization(data_x, data_y, segments, approx_models)
visualizer.visualize_results()

# 🔹 **Вывод времени выполнения**
end_time = time.time()
print(f"✅ Время выполнения: {end_time - start_time:.2f} секунд")
