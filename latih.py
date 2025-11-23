import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# --- KONFIGURASI ---
# Pastikan nama folder di sini SAMA PERSIS dengan struktur folder Anda
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/validation'

# Setting Gambar
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32
EPOCHS = 15  # Jumlah putaran belajar (bisa ditambah jika akurasi kurang)

# --- 1. PERSIAPAN DATA (IMAGE GENERATOR) ---
# Kita tambahkan variasi (augmentasi) agar model lebih pintar
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

print("Memuat data training...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary' # Binary karena folder hanya ada 2: Bergizi & Tidak
)

print("Memuat data validasi...")
validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# --- 2. MEMBANGUN MODEL CNN ---
model = Sequential([
    # Layer 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(2, 2),
    
    # Layer 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Layer 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dropout(0.5), # Biar tidak overfitting (terlalu menghafal)
    Dense(512, activation='relu'),
    
    # Output Layer: 1 neuron dengan Sigmoid (0 atau 1)
    Dense(1, activation='sigmoid')
])

# --- 3. KOMPILASI & TRAINING ---
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Mulai pelatihan... Mohon tunggu.")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# --- 4. SIMPAN MODEL ---
nama_file_model = 'model_makanan.h5'
model.save(nama_file_model)
print(f"SUKSES! Model telah disimpan sebagai {nama_file_model}")

# --- 5. CEK LABEL KELAS ---
# Penting untuk tahu 0 itu apa, 1 itu apa
labels = (train_generator.class_indices)
print("Label Kelas:", labels) 
# Biasanya: {'bergizi': 0, 'tidak_bergizi': 1} (sesuai abjad)