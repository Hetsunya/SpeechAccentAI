import pickle

# Загружаем словарь из файла
with open("country_count_dict.pkl", "rb") as f:
    country_count_dict = pickle.load(f)

print(country_count_dict)

import librosa
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Parameters
folder_path = "recordings"
N = 10  # Минимальное количество файлов для страны


# Получаем список файлов из папки
file_list = os.listdir(folder_path)

# Удаление файлов стран с количеством меньше N
for country, count in country_count_dict.items():
    if count < N:
        for filename in file_list:
            if country in filename:
                os.remove(os.path.join(folder_path, filename))
                print(f"Удален файл: {filename}")
