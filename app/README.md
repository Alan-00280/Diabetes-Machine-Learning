# 🩺 Aplikasi Prediksi Risiko Diabetes (FrontEnd Streamlit)

Aplikasi web interaktif untuk deteksi dini dan skrining risiko diabetes melitus berbasis **Machine Learning (Support Vector Machine / SVM)** yang dibangun menggunakan library **Streamlit**.

> **Mata Kuliah:** Praktikum Pembelajaran Mesin (UTS)  
> **Program Studi:** D4 Teknik Informatika, Fakultas Vokasi, Universitas Airlangga  

---

## 📌 Ringkasan Aplikasi

Aplikasi ini berfungsi sebagai sistem pendukung keputusan klinis (*clinical decision support system*) untuk membantu tenaga kesehatan atau pengguna umum melakukan penilaian risiko diabetes berdasarkan 8 parameter medis.

Model machine learning yang digunakan adalah **Support Vector Machine (SVM) dengan Kernel RBF** yang dilatih menggunakan strategi **Random Under-Sampling (RUS)**. Model ini dipilih karena memberikan performa terbaik dalam meminimalkan kasus *False Negative* (pasien diabetes yang tidak terdeteksi) dengan **Recall sebesar 76.47%** dan **F1-Score 68.42%** pada data uji independen.

---

## 🗂️ Struktur Direktori `app/`

```text
app/
├── main.py            # Skrip utama aplikasi web Streamlit
├── svm_bundle.pkl     # Paket lengkap model SVM + statistik skaler Z-Score
├── svm_model.pkl      # Objek model SVM terlatih (scikit-learn)
└── README.md          # Dokumentasi teknis aplikasi
```

### Penjelasan Artefak Model:
- **`svm_bundle.pkl`:** Berisi dictionary lengkap:
  - `model`: Objek classifier SVM terlatih (`CalibratedClassifierCV`).
  - `feature_names`: Urutan 8 kolom fitur medis.
  - `feature_mean` & `feature_std`: Nilai rata-rata dan deviasi standar dari data latih untuk normalisasi Z-score real-time pada data mentah yang diinputkan pengguna.
  - `class_labels`: Pemetaan label (`0: Tidak Diabetes`, `1: Diabetes`).
  - `metrics`: Ringkasan metrik evaluasi model.
- **`svm_model.pkl`:** Objek model SVM mandiri jika ingin diuji coba langsung tanpa bundle.

---

## ✨ Fitur-Fitur Utama Aplikasi

1. **Formulir Input Data Klinis Terstruktur (2 Kolom):**
   - **Profil Demografi & Fisik:** Usia, Jumlah Kehamilan, BMI (Indeks Massa Tubuh), dan Diabetes Pedigree Function (DPF).
   - **Pemeriksaan Laboratorium:** Glukosa Plasma 2 Jam (OGTT), Tekanan Darah Diastolik, Ketebalan Lipatan Kulit Trisep, dan Insulin Serum.
2. **Preset Simulasi Cepat (Quick Fill):**
   - Memungkinkan pengujian instan menggunakan profil preset:
     - *Pasien A (Risiko Rendah / Sehat)*
     - *Pasien B (Risiko Tinggi / Terindikasi Diabetes)*
     - *Pasien C (Kondisi Pre-Diabetes / Borderline)*
     - *Input Kustom / Manual*
3. **Normalisasi Z-Score Real-Time:**
   - Input angka asli (*raw values*) dari formulir otomatis distandardisasi menggunakan parameter skala yang sama persis dengan tahap pelatihan model (`train-diabetes.csv`).
4. **Hasil Prediksi & Indikator Probabilitas:**
   - Kartu status visual yang jelas:
     - 🟢 **Risiko Rendah / Tidak Diabetes**
     - 🔴 **Risiko Tinggi / Terindikasi Diabetes**
   - Bar persentase tingkat keyakinan (*confidence level*) model.
5. **Analisis Faktor Risiko Otomatis:**
   - Mendeteksi dan menyorot parameter spesifik yang melebihi batas aman (misal: Glukosa ≥ 140 mg/dL, Obesitas BMI ≥ 30, Hipertensi, dan Usia Rentan).
6. **Tab Evaluasi & Transparansi Ilmiah:**
   - Menampilkan perbandingan performa SVM Baseline vs SMOTE vs RUS beserta confusion matrix.
7. **Tab Panduan Nilai Klinis:**
   - Tabel referensi rentang normal, pre-diabetes, dan diabetes sesuai standar *American Diabetes Association* (ADA).

---

## 🚀 Panduan Menjalankan Aplikasi

### 1. Prasyarat Environment
Pastikan dependensi proyek telah terinstal (khususnya `streamlit`, `scikit-learn`, `pandas`, dan `numpy`).

Jika belum, instal dependensi melalui file `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### 2. Menjalankan Streamlit
Buka terminal / PowerShell di direktori utama proyek (`diabetes_uts/`), lalu jalankan perintah:

```powershell
streamlit run app/main.py
```

Setelah perintah dijalankan, antarmuka web akan terbuka secara otomatis di browser Anda pada alamat:
👉 **`http://localhost:8501`**

---

## 📊 Kinerja Model yang Diterapkan

Evaluasi model pada data uji independen ($N = 149$ baris):

| Skenario SVM | Akurasi | Presisi | Recall (Sensitivitas) | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Baseline (Original) | 73.83% | 67.65% | 45.10% | 54.12% | Terlalu banyak pasien luput |
| Oversampling (SMOTE) | 75.17% | 62.50% | 68.63% | 65.42% | Seimbang |
| **Undersampling (RUS)** | **75.84%** | **61.90%** | **76.47%** | **68.42%** | **Dipakai di Aplikasi (Terbaik)** |

---

## ⚠️ Catatan Medis & Disclaimer

Aplikasi ini dikembangkan untuk keperluan akademik dan penelitian tugas UTS Pembelajaran Mesin. Hasil prediksi yang dikeluarkan merupakan alat bantu skrining awal dan **bukan merupakan diagnosis medis mutlak**. Pasien tetap disarankan untuk berkonsultasi langsung dengan dokter atau fasilitas kesehatan profesional.
