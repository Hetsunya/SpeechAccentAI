import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import os

# --- Параметры ---
folder_path = "recordings"
labels_array = np.load("accent_labels_pad.npy")
label_encoder_classes = np.load("label_encoder_classes_pad.npy")
num_classes = len(np.unique(labels_array))

# --- Загрузка VGGish ---
vggish_model = hub.load("https://tfhub.dev/google/vggish/1")

# --- Функция для подготовки аудио ---
def prepare_audio(audio_data, target_length=10):
    """Обрезает или дополняет аудио до заданной длины."""
    audio_length = len(audio_data) / 22050  # Длина аудио в секундах
    if audio_length < target_length:
        # Дополнение тишиной
        padding_length = int((target_length - audio_length) * 22050)
        audio_data = np.pad(audio_data, (0, padding_length), 'constant')
    elif audio_length > target_length:
        # Обрезка
        audio_data = audio_data[:int(target_length * 22050)]
    return audio_data

# --- Функция для извлечения признаков VGGish ---
def extract_vggish_features(audio_data):
    # Преобразование аудиоданных в формат, подходящий для VGGish
    # (моно, 16 кГц, float32)
    audio_data = librosa.resample(audio_data, orig_sr=22050, target_sr=16000)
    audio_data = audio_data.astype(np.float32)
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Извлечение признаков VGGish
    features = vggish_model(audio_data)
    return features.numpy()

# --- Загрузка аудиоданных ---
audio_data = []
for filename in os.listdir(folder_path):
    if filename.endswith(".mp3"):
        print(filename)
        file_path = os.path.join(folder_path, filename)
        y, sr = librosa.load(file_path)
        audio_data.append(prepare_audio(y))

# --- Извлечение признаков VGGish ---
vggish_features = np.array([extract_vggish_features(x) for x in audio_data])

# --- Кодирование меток ---
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes
encoded_labels = label_encoder.transform(labels_array)

# --- Разделение данных ---
X_train, X_test, y_train, y_test = train_test_split(vggish_features, encoded_labels, test_size=0.2, random_state=42)

# --- Создание модели ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(vggish_features.shape[1], vggish_features.shape[2])),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(num_classes, activation='softmax')

])

# --- Компиляция модели ---
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- Early stopping ---
early_stopping = EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True)

# --- Обучение модели ---
# model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=1,
#           validation_data=(X_test, y_test), callbacks=[early_stopping])

def data_generator(features, labels, batch_size):
    while True:
        for i in range(0, len(features), batch_size):
            yield features[i:i+batch_size], labels[i:i+batch_size]

batch_size = 32  #  размер батча
train_generator = data_generator(X_train, y_train, batch_size)

model.fit(train_generator, epochs=20000, steps_per_epoch=len(X_train) // batch_size,
          validation_data=(X_test, y_test), callbacks=[early_stopping])

# --- Оценка модели ---
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Точность на тестовой выборке: {test_accuracy}")

# --- Сохранение модели ---
model.save("model.keras")
