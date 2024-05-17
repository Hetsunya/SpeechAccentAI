import librosa
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Parameters
folder_path = "recordings"

import os

# Parameters
folder_path = "recordings"


# Function to extract MFCCs and labels for a block of files
def extract_melspecs_and_labels_block(file_list):
    country_count = {}  # Словарь для хранения количества файлов для каждой страны
    labels_block = []
    for filename in file_list:
        if filename.endswith(".mp3"):
            country_info = filename.split('.')[0]
            country_name = ''.join(filter(str.isalpha, country_info))
            labels_block.append(country_name)
            # Увеличиваем счетчик для данной страны
            country_count[country_name] = country_count.get(country_name, 0) + 1
    return country_count, labels_block

# Получаем список файлов из папки
file_list = os.listdir(folder_path)

# Извлекаем MFCC и метки для всех файлов в блоке
country_count_dict, labels = extract_melspecs_and_labels_block(file_list)

print(country_count_dict)

import pickle


# Сохраняем словарь в файл
with open("country_count_dict.pkl", "wb") as f:
    pickle.dump(country_count_dict, f)



