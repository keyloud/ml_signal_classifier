# src/train_model.py

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from utils import load_labeled_signals
from feature_extraction import extract_features_batch

# Пути к файлам
data_paths = {
    "GSM": "dataset/r_gsm.txt",
    "WiFi": "dataset/r_wifi.txt",
    "LTE": "dataset/r_lte.txt",
    "Band40": "dataset/r_band40.txt"
}

# 1. Загружаем данные
X_raw, y_raw = load_labeled_signals(data_paths)

# 2. Извлекаем признаки
X = extract_features_batch(X_raw)  # [samples, features]
print(f"Размерность X_raw: {X_raw.shape}, длина y_raw: {len(y_raw)}")
print(f"Размерность X после извлечения признаков: {X.shape}")

# 3. Кодируем метки
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

# 4. Делим на train/test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 5. Обучаем модель (ограничим параметры: L2-регуляризация, макс. итерации)
model = LogisticRegression(
    penalty='l2',
    C=1.0,
    max_iter=1000,
    solver='liblinear'  # компактный solver
)

model.fit(X_train, y_train)

# 6. Оцениваем
y_pred = model.predict(X_val)
print("Отчёт по валидации:\n", classification_report(y_val, y_pred, target_names=encoder.classes_,zero_division=0))

# 7. Вычисление F1-Score для итоговой оценки
f1 = f1_score(y_val, y_pred, average='weighted')
print(f"Средний F1-Score: {f1}")

# 8. Получение количества параметров модели
n_params = model.coef_.size
print(f"Количество параметров модели: {n_params}")

# 9. Вычисляем итоговую оценку K
K = 100 * f1 + (4095 - n_params) / 400
print(f"Итоговая оценка K: {K}")

# 10. Сохраняем модель и энкодер
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/classifier.joblib")
joblib.dump(encoder, "models/label_encoder.joblib")

print("Модель успешно обучена и сохранена.")
