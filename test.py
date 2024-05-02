import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os

# Загрузка модели и LabelEncoder
model = tf.keras.models.load_model("model.h5")
label_encoder_classes = np.load("label_encoder_classes.npy")
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

folder_path = "recordings"

max_frames = 150  # Максимальное количество кадров MFCC
n_mfcc = 40  # Количество коэффициентов MFCC
hop_length = 512  # Размер окна
n_fft = 2048  # Размерность дискретизации


def predict_accent(audio_file):
    # Загрузка и предобработка аудио
    y, sr = librosa.load(folder_path+"/"+audio_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

    # Padding или усечение MFCCs
    if mfcc.shape[1] < max_frames:
        padded_mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode='constant', constant_values=0)
    else:
        padded_mfcc = mfcc[:, :max_frames]

    # Преобразование в формат, подходящий для модели
    mfcc_features = np.expand_dims(padded_mfcc, axis=0)
    mfcc_features = np.expand_dims(mfcc_features, axis=-1)  # Добавление размерности канала

    # Предсказание
    prediction = model.predict(mfcc_features)
    predicted_class_index = np.argmax(prediction)
    predicted_accent = label_encoder.classes_[predicted_class_index]

    return predicted_accent


for filename in os.listdir(folder_path):
    if filename.endswith(".mp3"):
        predicted_accent = predict_accent(filename)

        country_info = filename.split('.')[0]
        country_name = ''.join(filter(str.isalpha, country_info))

        print("Predicted accent:", predicted_accent)
        print("Real accent:", country_name)
