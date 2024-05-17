import librosa
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Parameters
folder_path = "recordings"
max_frames = 100
n_mfcc = 30
hop_length = 1024
n_fft = 4096
block_size = 100  # Number of files to process in each block

# Function to extract MFCCs and labels for a block of files
def extract_mfccs_and_labels_block(file_list):
    mfccs_block = []
    labels_block = []
    for filename in file_list:
        if filename.endswith(".mp3"):
            file_path = os.path.join(folder_path, filename)
            y, sr = librosa.load(file_path)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

            # Padding/truncation
            if mfcc.shape[1] < max_frames:
                padded_mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode='constant', constant_values=0)
            else:
                padded_mfcc = mfcc[:, :max_frames]

            mfccs_block.append(padded_mfcc)

            # Extract label from filename
            country_info = filename.split('.')[0]
            country_name = ''.join(filter(str.isalpha, country_info))
            labels_block.append(country_name)

    return np.array(mfccs_block), np.array(labels_block)

# Split files into blocks
files = os.listdir(folder_path)
file_blocks = [files[i:i+block_size] for i in range(0, len(files), block_size)]

# Process each block and accumulate data
all_mfccs = []
all_labels = []
for i, block in enumerate(file_blocks):
    print(f"Processing block {i+1}/{len(file_blocks)}")
    mfccs_block, labels_block = extract_mfccs_and_labels_block(block)
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
np.save("extracted_mfccs_pad.npy", extracted_mfccs)
np.save("accent_labels_pad.npy", accent_labels)
np.save("label_encoder_classes_pad.npy", label_encoder.classes_)
print("MFCCs, labels, and encoder classes saved!")
