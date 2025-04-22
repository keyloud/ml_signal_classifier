# src/predict.py

import os
import joblib
import numpy as np
from feature_extraction import extract_features_batch
from utils import load_unmarked_signals

# Загрузка обученной модели и энкодера
model = joblib.load("models/classifier.joblib")
encoder = joblib.load("models/label_encoder.joblib")

# Загрузка неразмеченных данных
unmarked_signals = load_unmarked_signals("dataset/unmarked.txt")

# Извлечение признаков для неразмеченных данных
X_unmarked = extract_features_batch(unmarked_signals)

# Убедимся, что X_unmarked — двумерный массив
print(f"Размерность X_unmarked: {X_unmarked.shape}")

# Получение предсказаний
y_pred = model.predict(X_unmarked)

# Декодирование предсказанных меток
y_pred_decoded = encoder.inverse_transform(y_pred)

# Запись предсказаний в файл predictions.txt
with open("predictions.txt", "w") as f:
    for label in y_pred_decoded:
        f.write(f"{label}\n")

print("Предсказания успешно записаны в predictions.txt.")
