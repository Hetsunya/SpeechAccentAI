import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder

mfccs_array = np.load("extracted_mfccs.npy")
labels_array = np.load("accent_labels.npy")
label_encoder_classes = np.load("label_encoder_classes.npy")
num_classes = len(np.unique(labels_array))

# Создание объекта LabelEncoder и загрузка сохраненных классов
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

# Преобразование строковых меток обратно в числовые значения
encoded_labels = label_encoder.transform(labels_array)


# Создание модели нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(mfccs_array.shape[1], mfccs_array.shape[2], 1)),  # Форма входных данных MFCC
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # num_classes - количество классов
])


# Компиляция модели с числовыми метками
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(mfccs_array, encoded_labels, epochs=100, batch_size=64, verbose=1)

model.save("accent_detection_model.h5")
