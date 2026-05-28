import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- 1. SETTINGS ---
SCRIPT_DIR = os.getcwd()
AUDIO_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "audio") # Point to your extracted sireNNet folders
CLASSES = ['ambulance', 'police', 'firetruck', 'traffic']
IMG_SIZE = (128, 128) # For spectrograms (optional) or use MFCCs

def extract_mfcc(file_path):
    """Converts audio file to a 1D 'fingerprint' of 40 features"""
    try:
        # Load 3 seconds of audio, resample to 22.05kHz
        audio, sr = librosa.load(file_path, duration=3.0, res_type='kaiser_fast')
        # Extract Mel-Frequency Cepstral Coefficients
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        # Average the features over time
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- 2. DATA PREPROCESSING ---
print("🔊 Extracting Audio Fingerprints (MFCCs)...")
features = []
labels = []

for idx, class_name in enumerate(CLASSES):
    class_folder = os.path.join(AUDIO_DATA_PATH, class_name)
    if not os.path.exists(class_folder):
        print(f"⚠️ Warning: Folder {class_name} not found!")
        continue
        
    for file in tqdm(os.listdir(class_folder), desc=f"Processing {class_name}"):
        if file.endswith('.wav'):
            file_path = os.path.join(class_folder, file)
            data = extract_mfcc(file_path)
            if data is not None:
                features.append(data)
                labels.append(idx)

X = np.array(features)
y = np.array(labels)

# Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. BUILD THE AUDIO BRAIN ---
print("🏗️ Building Audio Neural Network...")
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(40,)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(CLASSES), activation='softmax') # Output probability for each class
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- 4. TRAINING ---
print("🚀 Training Audio Model...")
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# --- 5. SAVE ---
save_path = os.path.join(SCRIPT_DIR, 'models', 'siren_model.h5')
model.save(save_path)
print(f"🎉 SUCCESS! Audio Model saved at: {save_path}")