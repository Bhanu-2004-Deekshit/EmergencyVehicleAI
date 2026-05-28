import os
import shutil
import tempfile
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# --- 1. BYPASS ONEDRIVE LOCKS ---
# We move the file to a system temp folder that OneDrive cannot monitor.
source_weights = r"C:\Users\sanab\OneDrive\Desktop\EmergencyVehicleAI\models\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
temp_dir = tempfile.gettempdir()
safe_weights_path = os.path.join(temp_dir, "local_weights_safe.h5")

if not os.path.exists(source_weights):
    print(f"❌ CRITICAL ERROR: File not found at {source_weights}")
    print("Please check if your file is named correctly in the 'models' folder.")
    exit()

print("🔄 Copying weights to a non-synced staging area...")
shutil.copy2(source_weights, safe_weights_path)
print(f"✅ Staging complete: {safe_weights_path}")

# --- 2. DATA LOADING ---
print("📊 Loading dataset...")
SCRIPT_DIR = os.getcwd()
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df['emergency_or_not'] = df['emergency_or_not'].astype(str)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_dataframe(
    df, directory=os.path.join(DATA_DIR, "train"), x_col='image_names', y_col='emergency_or_not',
    target_size=(224, 224), batch_size=32, class_mode='binary', subset='training'
)

# --- 3. THE "DEEP LOAD" FIX ---
# Instead of loading weights inside the function, we build a blank 'Skeleton' first.
print("🏗️ Building model architecture (Skeleton)...")
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None # Crucial: Don't load yet to avoid the Errno 22 lock
)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Now we manually 'pour' the knowledge into the skeleton.
print("💉 Injecting weights into the skeleton...")
try:
    model.layers[0].load_weights(safe_weights_path)
    print("✅ Weights injected successfully!")
except Exception as e:
    print(f"❌ Failed to load weights: {e}")
    exit()

# --- 4. TRAINING ---
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("🚀 Starting training...")
# Note: Using 5 epochs and 20 steps per epoch so you can see results quickly.
model.fit(train_gen, epochs=5, steps_per_epoch=20)

# --- 5. SAVING ---
save_path = os.path.join(SCRIPT_DIR, 'models', 'emergency_vehicle_model.h5')
model.save(save_path)
print(f"🎉 SUCCESS! Model saved at: {save_path}")