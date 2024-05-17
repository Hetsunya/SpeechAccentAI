import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
import os

# --- Параметры ---
model_path = "model.keras"  # Путь к сохраненной модели
label_encoder_classes = np.load("label_encoder_classes_pad.npy")  # Путь к классам кодировщика меток
test_folder = "test"  # Путь к папке с тестовыми аудиофайлами

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
    audio_data = librosa.resample(audio_data, orig_sr=22050, target_sr=16000)
    audio_data = audio_data.astype(np.float32)
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Извлечение признаков VGGish
    features = vggish_model(audio_data)
    return features.numpy()

def predict_accent(audio_file_path):
    # Загрузка аудиофайла
    y, sr = librosa.load(audio_file_path)

    # Подготовка аудио
    y = prepare_audio(y)

    # Извлечение признаков VGGish
    features = extract_vggish_features(y)
    features = np.expand_dims(features, axis=0)  # Добавление размерности для батча

    # Предсказание модели
    predictions = model.predict(features)
    predicted_class_index = np.argmax(predictions)

    # Декодирование метки
    predicted_accent = label_encoder.inverse_transform([predicted_class_index])[0]

    return predicted_accent

# --- Загрузка модели ---
model = tf.keras.models.load_model(model_path)

# --- Декодирование меток ---
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

# --- Предсказания для каждого файла в папке test ---
correct_predictions = 0
total_predictions = 0

for filename in os.listdir(test_folder):
    if filename.endswith(".mp3"):
        file_path = os.path.join(test_folder, filename)

        # Извлечение истинной метки из имени файла
        true_accent = filename.split('_')[0]

        # Предсказание акцента
        predicted_accent = predict_accent(file_path)

        # Сравнение предсказания с истинной меткой
        if predicted_accent == true_accent:
            correct_predictions += 1
        total_predictions += 1

        print(f"Файл: {filename}, Истинный акцент: {true_accent}, Предсказанный акцент: {predicted_accent}")

# --- Вычисление точности ---
accuracy = (correct_predictions / total_predictions) * 100
print(f"\nТочность: {accuracy:.2f}%")
