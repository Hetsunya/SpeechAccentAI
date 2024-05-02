import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os

folder_path = "recordings"

mfccs = []
max_frames = 150
mfccs = []
labels = []

print("Подготовка файлов")

for filename, i in zip(os.listdir(folder_path), range(len(os.listdir(folder_path)))):
    if filename.endswith(".mp3"):
        print(f"Подгоовка файла {i+1}")

        country_info = filename.split('.')[0]  # Пример: italian11.mp3 -> 'italian11'
        country_name = ''.join(filter(str.isalpha, country_info))  # Получаем только слово из имени файла italian11 -> italian
        print(country_name)

        # Загрузка аудиофайла и извлечение MFCC
        file_path = os.path.join(folder_path, filename)
        y, sr = librosa.load(file_path)

         # Параметры для увеличения детализации MFCC
        n_mfcc = 40  # Количество коэффициентов MFCC
        hop_length = 512  # Размер окна
        n_fft = 2048  # Размерность дискретизации

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length= hop_length, n_fft=n_fft)

        # Выравнивание MFCC до одинаковой длины, если с этой хуйней не будет робить, то нужно сделать .T
        if mfcc.shape[1] < max_frames:
            padded_mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode='constant', constant_values=0)
        else:
            padded_mfcc = mfcc[:, :max_frames]

        mfccs.append(padded_mfcc)
        labels.append(country_name)

# Преобразование списков в массивы numpy
mfccs_array = np.array(mfccs)
labels_array = np.array(labels)

# Создание объекта LabelEncoder
label_encoder = LabelEncoder()

# Преобразование строковых меток в числовые значения
encoded_labels = label_encoder.fit_transform(labels_array)


np.save("extracted_mfccs.npy", mfccs_array)
np.save("accent_labels.npy", labels_array)
np.save("label_encoder_classes.npy", label_encoder.classes_)
print(label_encoder.classes_)
print(label_encoder)
print(encoded_labels)

print("подготовка файлов завершина")

# Определение количества уникальных классов
num_classes = len(np.unique(labels_array))
print("Количество уникальных классов:", num_classes)
