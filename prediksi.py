import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import sys

# --- KONFIGURASI ---
MODEL_PATH = 'model_makanan.h5'

# Cek apakah file model sudah ada
try:
    model = load_model(MODEL_PATH)
    print("Model berhasil dimuat!")
except:
    print("Error: File model_makanan.h5 tidak ditemukan.")
    print("Jalankan 'latih.py' terlebih dahulu!")
    sys.exit()

def prediksi_gambar(path_gambar):
    try:
        # Load dan atur ukuran gambar
        img = image.load_img(path_gambar, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0 # Normalisasi

        # Prediksi
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        score = classes[0][0]

        print(f"\nFile: {path_gambar}")
        print(f"Skor Prediksi: {score:.4f}")
        
        # LOGIKA HASIL (Sesuaikan dengan output label di latih.py)
        # Jika bergizi = 0 dan tidak_bergizi = 1:
        if score > 0.5:
            print("Keputusan: MAKANAN TIDAK BERGIZI")
        else:
            print("Keputusan: MAKANAN BERGIZI")
            
    except Exception as e:
        print(f"Error saat memproses gambar: {e}")
        print("Pastikan path/nama file gambar benar.")

# --- CARA PAKAI ---
# Ganti 'tes.jpg' dengan nama file gambar yang mau Anda tes
# Pastikan file gambar tersebut ada di folder proyek Anda
file_tes = 'tes1.jpg' 

prediksi_gambar(file_tes)