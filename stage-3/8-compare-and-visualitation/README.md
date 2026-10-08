# Stage 3 - Compare and Visualisation

## Pemilik
Luthfi Alan Perdana (NIM: 434241052)

## Deskripsi Tahap
Tahap ini merupakan tahap akhir dari Stage 3 (Evaluasi dan Visualisasi) yang bertugas melakukan perbandingan komprehensif terhadap performa model Decision Tree di bawah tiga skenario penanganan dataset:
1. **Skenario 1 (Baseline)**: Tanpa resampling (distribusi kelas asli yang tidak seimbang).
2. **Skenario 2 (Oversampling)**: Penyeimbangan kelas menggunakan *Synthetic Minority Over-sampling Technique* (SMOTE).
3. **Skenario 3 (Undersampling)**: Penyeimbangan kelas menggunakan *Random Under Sampling* (RUS).

Tahap ini menghasilkan visualisasi komparatif berstandar publikasi serta rekapitulasi data metrik untuk mendukung analisis evaluasi, perbandingan antarskenario, dan pembahasan kinerja model.

---

## Input
- `stage-3/7-confusion-matrix/output/evaluation-metrics.csv`: Data metrik evaluasi dan matriks konfusi dari ketiga skenario.
- `stage-2/6-decision-tree-and-testing/output/test-predictions.csv`: Data prediksi 149 sampel uji dari masing-masing model beserta label aktual.

---

## Metode dan Metrik Evaluasi
1. **Confusion Matrix Analysis**:
   - Menghitung persentase terhadap kelas aktual pada setiap kuadran (True Negative, False Positive, False Negative, True Positive).
   - Menganalisis implikasi klinis diagnosis medis pada setiap jenis kesalahan (khususnya bahaya kritis dari *False Negative*).
2. **Metrik Komparasi Utama**:
   - Akurasi: $\frac{\text{TN} + \text{TP}}{\text{Total}}$
   - Presisi: $\frac{\text{TP}}{\text{TP} + \text{FP}}$
   - Recall (Sensitivitas): $\frac{\text{TP}}{\text{TP} + \text{FN}}$
   - F1-Score: $2 \times \frac{\text{Presisi} \times \text{Recall}}{\text{Presisi} + \text{Recall}}$
3. **Analisis Kesalahan Medis**:
   - Alarm Palsu / FP: Sehat namun diprediksi sakit.
   - Luput / FN: Sakit namun diprediksi sehat (kesalahan berisiko tinggi).
4. **Radar Multi-Metric Analysis**:
   - Memetakan trade-off antara Akurasi, Presisi, Recall, F1-Score, dan Spesifisitas.

---

## Cara Menjalankan
Pastikan *virtual environment* Python sudah aktif, lalu jalankan dari root direktori proyek (`diabetes_uts`):

```bash
python stage-3/8-compare-and-visualitation/compare-and-visualisation.py
```

---

## Output
Semua artefak visualisasi dan data disimpan di folder `output/`:
1. `gambar-4-15-cm-baseline.png` — Visualisasi individual Confusion Matrix Skenario 1 (Baseline) lengkap dengan persentase baris dan metrik.
2. `gambar-4-16-cm-oversampling.png` — Visualisasi individual Confusion Matrix Skenario 2 (Oversampling SMOTE).
3. `gambar-4-17-cm-undersampling.png` — Visualisasi individual Confusion Matrix Skenario 3 (Undersampling RUS).
4. `gambar-4-18-perbandingan-kinerja.png` — Gambar 4.18 berupa multi-panel visualisasi yang menggabungkan grafik perbandingan 4 metrik evaluasi dan grafik analisis kesalahan diagnosis medis (FP vs FN).
5. `perbandingan-metrik.png` — Salinan grafik perbandingan metrik evaluasi.
6. `perbandingan-radar.png` — Radar chart trade-off performa multivariat.
7. `perbandingan-skenario.csv` — File CSV yang memuat seluruh nilai kuadran CM, metrik utama, spesifisitas, FPR, dan FNR tiap skenario.

---

## Ringkasan Hasil Kinerja

| Skenario | Akurasi | Presisi | Recall | F1-Score | FP (Alarm Palsu) | FN (Luput) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Original) | 77.18% | 75.76% | 49.02% | 59.52% | 8 pasien | 26 pasien |
| Oversampling (SMOTE) | 69.80% | 54.41% | 72.55% | 62.18% | 31 pasien | 14 pasien |
| **Undersampling (RUS)** | **74.50%** | **59.42%** | **80.39%** | **68.33%** | **28 pasien** | **10 pasien** |

**Kesimpulan Penentuan Skenario Terbaik:**  
**Skenario 3 (Undersampling / RUS)** merupakan skenario terbaik karena menghasilkan **Recall tertinggi (80.39%)** dan **F1-Score tertinggi (68.33%)**, yang berhasil memangkas jumlah pasien diabetes yang luput dari 26 orang (baseline) menjadi hanya 10 orang. Dalam domain medis, kemampuan mendeteksi pasien yang sakit secara dini (meminimalkan *False Negative*) jauh lebih diprioritaskan daripada sekadar mengejar akurasi umum.
