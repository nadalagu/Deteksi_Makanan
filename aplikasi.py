import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import threading # Agar UI tidak hang saat loading model

# --- KONFIGURASI TEMA ---
ctk.set_appearance_mode("Dark")  # Pilihan: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Pilihan: "blue", "green", "dark-blue"

class ModernApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Konfigurasi Window Utama
        self.title("Smart Nutrition Detector")
        self.geometry("900x600")
        
        # Grid Layout (2 Kolom)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- LOAD MODEL (Di Thread terpisah agar aplikasi cepat terbuka) ---
        self.model = None
        self.model_path = "model_makanan.h5"
        self.load_model_thread()

        # --- VARIABEL ---
        self.cap = None
        self.is_camera_on = False

        # --- BAGIAN KIRI: SIDEBAR MENU ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # Judul di Sidebar
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="NUTRI-SCAN AI", 
                                       font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.desc_label = ctk.CTkLabel(self.sidebar_frame, text="Deteksi Makanan\nBergizi vs Tidak", 
                                      font=ctk.CTkFont(size=12))
        self.desc_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        # Tombol-tombol
        self.btn_kamera = ctk.CTkButton(self.sidebar_frame, text="Mulai Kamera", 
                                        command=self.toggle_camera, height=40,
                                        fg_color="#dbdbdb", text_color="black", hover_color="#c9c9c9")
        self.btn_kamera.grid(row=2, column=0, padx=20, pady=10)

        self.btn_upload = ctk.CTkButton(self.sidebar_frame, text="Upload File", 
                                        command=self.upload_gambar, height=40,
                                        fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        self.btn_upload.grid(row=3, column=0, padx=20, pady=10)

        # Tombol Keluar
        self.btn_keluar = ctk.CTkButton(self.sidebar_frame, text="Keluar Aplikasi", 
                                        command=self.destroy, fg_color="#C0392B", hover_color="#E74C3C")
        self.btn_keluar.grid(row=5, column=0, padx=20, pady=20)


        # --- BAGIAN KANAN: DISPLAY AREA ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        # Kotak Tampilan Gambar/Video
        self.video_frame = ctk.CTkFrame(self.main_frame, fg_color="#2b2b2b", corner_radius=15)
        self.video_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Kamera Mati / Belum Ada Gambar", text_color="gray")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Kotak Hasil Prediksi
        self.result_frame = ctk.CTkFrame(self.main_frame, height=100, corner_radius=10)
        self.result_frame.pack(fill="x", padx=10, pady=(10, 0))

        self.lbl_result_title = ctk.CTkLabel(self.result_frame, text="HASIL ANALISA:", 
                                             font=ctk.CTkFont(size=14))
        self.lbl_result_title.pack(pady=(10,0))

        self.lbl_prediction = ctk.CTkLabel(self.result_frame, text="-", 
                                           font=ctk.CTkFont(size=28, weight="bold"))
        self.lbl_prediction.pack(pady=(0,10))

        # Mulai loop video
        self.video_loop()

    def load_model_thread(self):
        """Load model di background agar UI tidak macet"""
        def _load():
            try:
                self.model = load_model(self.model_path)
                print("Model berhasil dimuat.")
                self.lbl_prediction.configure(text="Sistem Siap!")
            except:
                self.lbl_prediction.configure(text="Error: Model tidak ditemukan!")
        
        threading.Thread(target=_load, daemon=True).start()

    def prediksi(self, img_array):
        if self.model is None: return "Tunggu Model...", "gray"

        img_array = img_array / 255.0
        pred = self.model.predict(img_array, verbose=0)
        score = pred[0][0]

        if score > 0.5:
            text = f"TIDAK BERGIZI ({score*100:.1f}%)"
            color = "#E74C3C" # Merah Modern
        else:
            text = f"BERGIZI ({(1-score)*100:.1f}%)"
            color = "#2ECC71" # Hijau Modern
        
        return text, color

    def video_loop(self):
        if self.is_camera_on and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # 1. Konversi untuk Display (CTK Image)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Agar gambar tidak gepeng, kita pakai PIL Image
                pil_img = Image.fromarray(frame_rgb)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(500, 375))
                
                self.video_label.configure(image=ctk_img, text="")

                # 2. Prediksi
                img_input = cv2.resize(frame_rgb, (150, 150))
                img_input = image.img_to_array(img_input)
                img_input = np.expand_dims(img_input, axis=0)
                
                hasil, warna = self.prediksi(img_input)
                self.lbl_prediction.configure(text=hasil, text_color=warna)

        self.after(10, self.video_loop)

    def toggle_camera(self):
        if self.is_camera_on:
            self.is_camera_on = False
            if self.cap: self.cap.release()
            self.video_label.configure(image=None, text="Kamera Dimatikan")
            self.btn_kamera.configure(text="Mulai Kamera", fg_color="#dbdbdb", text_color="black")
        else:
            self.cap = cv2.VideoCapture(0)
            self.is_camera_on = True
            self.btn_kamera.configure(text="Stop Kamera", fg_color="#E67E22", text_color="white") # Oranye

    def upload_gambar(self):
        if self.is_camera_on: self.toggle_camera()

        file_path = filedialog.askopenfilename()
        if file_path:
            # Tampilkan Gambar
            pil_img = Image.open(file_path)
            # Resize proporsional agar muat di layar
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(500, 400))
            self.video_label.configure(image=ctk_img, text="")

            # Prediksi
            img_input = image.load_img(file_path, target_size=(150, 150))
            img_input = image.img_to_array(img_input)
            img_input = np.expand_dims(img_input, axis=0)
            
            hasil, warna = self.prediksi(img_input)
            self.lbl_prediction.configure(text=hasil, text_color=warna)

if __name__ == "__main__":
    app = ModernApp()
    app.mainloop()