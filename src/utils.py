# src/utils.py

import numpy as np

def load_signals_from_txt(path: str) -> np.ndarray:
    """
    Загружает сигналы из .txt файла.
    Каждый сигнал — строка из 1024 вещественных чисел, разделённых пробелами.
    Возвращает 2D numpy массив: [num_samples, 1024]
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    
    signals = [np.array(list(map(float, line.strip().split()))) for line in lines]
    return np.array(signals)


def load_labeled_signals(paths: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Принимает словарь: {label_name: path_to_file}, загружает все сигналы
    и возвращает X (сигналы), y (метки).
    """
    all_signals = []
    all_labels = []

    for label, path in paths.items():
        signals = load_signals_from_txt(path)
        all_signals.append(signals)
        all_labels.extend([label] * len(signals))
    
    X = np.vstack(all_signals)
    y = np.array(all_labels)
    return X, y

def load_unmarked_signals(file_path):
    """
    Загружает неразмеченные сигналы из файла.
    """
    data = np.loadtxt(file_path)
    return data