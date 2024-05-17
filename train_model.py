import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

mfccs_array = np.load("extracted_mfccs_pad.npy")
labels_array = np.load("accent_labels_pad.npy")
label_encoder_classes = np.load("label_encoder_classes_pad.npy")
num_classes = len(np.unique(labels_array))

# Создание объекта LabelEncoder и загрузка сохраненных классов
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

# Преобразование строковых меток обратно в числовые значения
encoded_labels = label_encoder.transform(labels_array)

# Разделение выборки на обучающую и тестовую
X_train, X_test, y_train, y_test = train_test_split(mfccs_array, encoded_labels, test_size=0.2, random_state=42)

# Создание модели нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(mfccs_array.shape[1], mfccs_array.shape[2], 1)),
    tf.keras.layers.Conv2D(8, (2, 2), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.9),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Компиляция модели с числовыми метками
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели на обучающей выборке
model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=1)

# Оценка производительности модели на тестовой выборке
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Точность на тестовой выборке: {test_accuracy}")
model.save("model.h5")
