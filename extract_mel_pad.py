import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# Параметры
folder_path = "recordings"
mels_folder = "mels"
max_frames = 125
n_mels = 64
hop_length = 1024
n_fft = 4096
block_size = 100


# --- Аугментация ---
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.7),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.6),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.6),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.6),
])

from pydub import AudioSegment

# Функция для загрузки аудиофайла MP3 и конвертации его в WAV
def load_mp3_and_convert_to_wav(file_path):
    audio = AudioSegment.from_mp3(file_path)
    # Конвертировать в 16-битный WAV
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_file_path = file_path.replace(".mp3", ".wav")
    audio.export(wav_file_path, format="wav")
    return wav_file_path


# Функция для извлечения Mel и labels для блока файлов с аугментацией
def extract_melspecs_and_labels_block(file_list):
    melspecs_block = []
    labels_block = []
    for filename in file_list:
        if filename.endswith(".mp3"):
            file_path = os.path.join(folder_path, filename)
            # Конвертировать MP3 в WAV
            wav_file_path = load_mp3_and_convert_to_wav(file_path)
            y, sr = librosa.load(wav_file_path)

            # --- Применение аугментаций ---
            augmented_data = augment(y, sample_rate=sr)
            # --- Конец аугментации ---

            # Вычисление мел-спектрограммы
            melspec = librosa.feature.melspectrogram(y=augmented_data, sr=sr, n_mels=n_mels,
                                                    hop_length=hop_length, n_fft=n_fft)

            melspec_db = librosa.power_to_db(melspec, ref=np.max)

            if melspec_db.shape[1] < max_frames:
                padded_melspec = np.pad(melspec_db, ((0, 0), (0, max_frames - melspec_db.shape[1])),
                                        mode='constant', constant_values=-80.0)
            else:
                padded_melspec = melspec_db[:, :max_frames]
            melspecs_block.append(padded_melspec)

            country_info = filename.split('.')[0]
            country_name = ''.join(filter(str.isalpha, country_info))
            labels_block.append(country_name)

            # --- Сохранение визуализации ---
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(padded_melspec, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel-спектрограмма: {filename}')

            # Создание папки mels, если ее нет
            os.makedirs(mels_folder, exist_ok=True)

            plt.savefig(os.path.join(mels_folder, f'{country_name}_{filename[:-4]}.png'))
            plt.close()

    return np.array(melspecs_block), np.array(labels_block)


# Split files into blocks
files = os.listdir(folder_path)
file_blocks = [files[i:i+block_size] for i in range(0, len(files), block_size)]

# Process each block and accumulate data
all_mfccs = []
all_labels = []
for i, block in enumerate(file_blocks):
    print(f"Processing block {i+1}/{len(file_blocks)}")
    mfccs_block, labels_block = extract_melspecs_and_labels_block(block)
    all_mfccs.append(mfccs_block)
    all_labels.append(labels_block)
    del mfccs_block, labels_block  # Delete block data to free memory

# Combine data from all blocks
extracted_mfccs = np.concatenate(all_mfccs, axis=0)
accent_labels = np.concatenate(all_labels, axis=0)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(accent_labels)

# Save extracted data
np.save("extracted_mel_pad.npy", extracted_mfccs)
np.save("accent_labels_pad.npy", accent_labels)
np.save("label_encoder_classes_pad.npy", label_encoder.classes_)
print("mel, labels, and encoder classes saved!")
