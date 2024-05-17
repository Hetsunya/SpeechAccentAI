import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os

# Загрузка модели и LabelEncoder
model = tf.keras.models.load_model("model.h5")
label_encoder_classes = np.load("label_encoder_classes_pad.npy")
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

folder_path = "test"

max_frames = 100
n_mfcc = 50
hop_length = 64
n_fft = 4096
block_size = 100  # Number of files to process in each block
 # Number of files to process in each block

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

count = 0
for filename in os.listdir(folder_path):
    if filename.endswith(".mp3"):
        predicted_accent = predict_accent(filename)

        country_info = filename.split('.')[0]
        country_name = ''.join(filter(str.isalpha, country_info))
        if(predicted_accent == country_name):
            count += 1
        else:
            print("Predicted accent:", predicted_accent)
            print("Real accent:", country_name)

print(count / len(os.listdir(folder_path)) * 100, "%")
