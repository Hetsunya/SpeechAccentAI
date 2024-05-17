import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping

mfccs_array = np.load("extracted_mel_pad.npy")
print(mfccs_array)
labels_array = np.load("accent_labels_pad.npy")
label_encoder_classes = np.load("label_encoder_classes_pad.npy")
print(label_encoder_classes)
num_classes = len(np.unique(labels_array))

label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

encoded_labels = label_encoder.transform(labels_array)

X_train, X_test, y_train, y_test = train_test_split(mfccs_array, encoded_labels, test_size=0.1, random_state=42)
print(X_train, X_test, y_train, y_test)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(mfccs_array.shape[1], mfccs_array.shape[2], 1)),
    tf.keras.layers.Reshape((mfccs_array.shape[1], mfccs_array.shape[2], 1, 1)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 1))),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (2, 2), activation='relu', padding='same')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 1))),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(8, activation='relu'),  # Добавлен второй полносвязный слой
    tf.keras.layers.Dropout(0.6),  # Увеличено значение Dropout
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, epochs=25, batch_size=8, verbose=1,
          validation_data=(X_test, y_test), callbacks=[early_stopping])

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Точность на тестовой выборке: {test_accuracy}")
model.save("model.keras", overwrite=True)
