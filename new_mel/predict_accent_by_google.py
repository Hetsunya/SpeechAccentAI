import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder

# --- Параметры ---
model_path = "model.keras"  # Путь к сохраненной модели
label_encoder_classes = np.load("label_encoder_classes_pad.npy")  # Путь к классам кодировщика меток

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

# --- Загрузка модели ---
model = tf.keras.models.load_model(model_path)

# --- Декодирование меток ---
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

# --- Функция для предсказания акцента ---
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

# --- Пример использования ---
audio_file = "Recording from 2024-05-03 19.22.50.mp3"  # Путь к аудиофайлу для предсказания
predicted_accent = predict_accent(audio_file)

print(f"Предсказанный акцент: {predicted_accent}")
