import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load model and LabelEncoder
model = tf.keras.models.load_model("model.h5")
label_encoder_classes = np.load("label_encoder_classes_pad.npy")
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes


test_folder = "test"  # Path to the folder containing test audio files


max_frames = 125
n_mels = 64
hop_length = 1024
n_fft = 4096

def predict_accent(audio_file):
    # Load and preprocess audio
    y, sr = librosa.load(audio_file)

    # Compute mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Padding or truncation
    if mel_spectrogram_db.shape[1] < max_frames:
        padded_mel_spectrogram = np.pad(mel_spectrogram_db, ((0, 0), (0, max_frames - mel_spectrogram_db.shape[1])), mode='constant', constant_values=-80.0)
    else:
        padded_mel_spectrogram = mel_spectrogram_db[:, :max_frames]

    # Reshape for model input
    mel_features = np.expand_dims(padded_mel_spectrogram, axis=0)
    mel_features = np.expand_dims(mel_features, axis=-1)  # Add channel dimension

    # Predict
    prediction = model.predict(mel_features)
    predicted_class_index = np.argmax(prediction)
    predicted_accent = label_encoder.classes_[predicted_class_index]

    return predicted_accent

# Predict for all files in the test folder
for filename in os.listdir(test_folder):
    print(filename)
    if filename.endswith(".mp3"):
        predicted_accent = predict_accent(filename)
        print(f"File: {filename}, Predicted Accent: {predicted_accent}")
