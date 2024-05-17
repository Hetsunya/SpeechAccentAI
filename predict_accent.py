import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load model and LabelEncoder
model = tf.keras.models.load_model("best_CNN.h5")  # Assuming model.h5 is in the same directory
label_encoder_classes = np.load("label_encoder_classes_pad.npy")
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

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

    # Reshape for model input - Apply the reshaping here
    mel_features = np.expand_dims(padded_mel_spectrogram, axis=0)
    mel_features = tf.reshape(mel_features, (1, mel_features.shape[1], mel_features.shape[2], 1)) # Reshape to 4D

    # Predict
    prediction = model.predict(mel_features)
    predicted_class_index = np.argmax(prediction)
    predicted_accent = label_encoder.classes_[predicted_class_index]

    return predicted_accent

# Example usage:
audio_file_path = "Recording from 2024-05-03 19.50.23.mp3"  # Replace with your audio file path
predicted_accent = predict_accent(audio_file_path)
print("Predicted Accent:", predicted_accent)
