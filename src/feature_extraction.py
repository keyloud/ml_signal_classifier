# src/feature_extraction.py

import numpy as np

def extract_features(signal: np.ndarray) -> np.ndarray:
    """
    Извлекает признаки из одного сигнала (1D-массив из 1024 значений).
    Возвращает вектор признаков (1D-массив).
    """
    features = []

    # Статистические признаки
    features.append(np.mean(signal))              # Среднее
    features.append(np.std(signal))               # Стандартное отклонение
    features.append(np.min(signal))               # Минимум
    features.append(np.max(signal))               # Максимум
    features.append(np.median(signal))            # Медиана
    features.append(np.percentile(signal, 25))    # 1-й квартиль
    features.append(np.percentile(signal, 75))    # 3-й квартиль

    # Энергия сигнала
    features.append(np.sum(signal ** 2))

    # Признаки из преобразования Фурье
    fft = np.fft.fft(signal)
    fft_magnitude = np.abs(fft)[:len(fft)//2]  # Используем только положительные частоты

    # Частотные признаки
    features.append(np.mean(fft_magnitude))
    features.append(np.std(fft_magnitude))
    features.append(np.max(fft_magnitude))
    features.append(np.sum(fft_magnitude ** 2))  # Энергия в частотной области

    # Вернем как numpy массив
    return np.array(features)


def extract_features_batch(signals: np.ndarray) -> np.ndarray:
    """
    Принимает массив сигналов (2D: [samples, 1024]).
    Возвращает массив признаков [samples, features].
    """
    return np.array([extract_features(signal) for signal in signals])
