# Final App: Pipeline End-to-End Klasifikasi Diabetes

Folder ini mengintegrasikan seluruh tahapan pemrosesan data dan pemodelan machine learning dari **Stage 1**, **Stage 2**, hingga **Stage 3** ke dalam satu skrip Python yang ringkas dan mandiri.

## Isi File

- [`main.py`](main.py): Skrip utama yang menjalankan pipeline lengkap dari data mentah hingga evaluasi.
- `output/`: Direktori penyimpanan output hasil eksekusi.
  - `evaluation-summary.csv`: Ringkasan metrik evaluasi (Akurasi, Presisi, Recall, F1-Score, dan Confusion Matrix).
  - `evaluation-visualization.png`: Visualisasi gabungan matriks kebingungan dan perbandingan metrik antarskenario.

## Alur Pemrosesan di `main.py`

1. **Stage 1 (Pembersihan & Pembagian Data)**:
   - Imputasi nilai 0 pada kolom medis (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) menggunakan median kolom.
   - Penghapusan baris duplikat.
   - Penanganan outlier menggunakan metode IQR Capping (*Winsorization*).
   - Pembagian data (*Stratified Split* 80% train, 20% test, `random_state=42`).

2. **Stage 2 (Transformasi, Resampling & Pelatihan Model)**:
   - Z-score scaling dihitung hanya dari data training untuk mencegah *data leakage*.
   - Resampling data training:
     - Baseline (data training asli).
     - Oversampling menggunakan SMOTE.
     - Undersampling menggunakan Random Under Sampler (RUS).
   - Pelatihan Decision Tree (`criterion="entropy"`, `max_depth=5`, `random_state=42`).

3. **Stage 3 (Evaluasi & Visualisasi)**:
   - Pengujian terhadap data uji (*test set*).
   - Penghitungan Confusion Matrix, Akurasi, Presisi, Recall, dan F1-Score.
   - Penyimpanan tabel metrik dan plot visualisasi komparatif.

## Cara Menjalankan

Dari direktori root repositori:

```powershell
.\.venv\Scripts\python final-app/main.py
```

Atau masuk ke direktori `final-app/`:

```powershell
cd final-app
..\.venv\Scripts\python main.py
```
