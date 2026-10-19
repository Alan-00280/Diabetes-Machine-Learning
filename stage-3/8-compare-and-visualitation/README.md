# Stage 3 - Compare and Visualisation

## Pemilik
Luthfi Alan Perdana (NIM: 434241052)

## Deskripsi Tahap
Tahap ini merupakan tahap akhir dari Stage 3 (Evaluasi dan Visualisasi) yang bertugas melakukan perbandingan komprehensif terhadap performa tiga algoritma pembelajaran mesin:
1. **Decision Tree** (Tahap 7-1)
2. **Naive Bayes** (Tahap 7-2)
3. **Support Vector Machine (SVM)** (Tahap 7-3)

Masing-masing model dievaluasi di bawah tiga skenario penanganan dataset tidak seimbang (*imbalanced data*):
- **Skenario 1 (Baseline)**: Tanpa resampling (distribusi kelas asli yang tidak seimbang).
- **Skenario 2 (Oversampling)**: Penyeimbangan kelas menggunakan *Synthetic Minority Over-sampling Technique* (SMOTE).
- **Skenario 3 (Undersampling)**: Penyeimbangan kelas menggunakan *Random Under Sampling* (RUS).

Total **9 konfigurasi model-skenario** dibandingkan secara menyeluruh untuk mengidentifikasi model klasifikasi dan teknik resampling paling optimal untuk deteksi diabetes melitus.

---

## Input
- `stage-3/7-1-confusion-matrix-decision-tree/output/evaluation-metrics.csv`
- `stage-3/7-2-confusion-matrix-naive-bayes/output/evaluation-metrics.csv`
- `stage-3/7-3-confusion-matrix-svm/output/evaluation-metrics.csv`

---

## Metode dan Metrik Evaluasi
1. **Confusion Matrix Analysis**:
   - Menghitung persentase terhadap kelas aktual pada setiap kuadran (True Negative, False Positive, False Negative, True Positive).
   - Menganalisis implikasi klinis diagnosis medis pada setiap jenis kesalahan (khususnya bahaya kritis dari *False Negative* / luput).
2. **Metrik Komparasi Utama**:
   - Akurasi: $\frac{\text{TN} + \text{TP}}{\text{Total}}$
   - Presisi: $\frac{\text{TP}}{\text{TP} + \text{FP}}$
   - Recall (Sensitivitas): $\frac{\text{TP}}{\text{TP} + \text{FN}}$
   - F1-Score: $2 \times \frac{\text{Presisi} \times \text{Recall}}{\text{Presisi} + \text{Recall}}$
   - Spesifisitas: $\frac{\text{TN}}{\text{TN} + \text{FP}}$
3. **Analisis Kesalahan Medis**:
   - Alarm Palsu / FP: Sehat namun diprediksi sakit.
   - Luput / FN: Sakit namun diprediksi sehat (kesalahan berisiko fatal).
4. **Radar Multi-Metric Analysis**:
   - Memetakan trade-off antara Akurasi, Presisi, Recall, F1-Score, dan Spesifisitas antar-algoritma terbaik.

---

## Cara Menjalankan
Pastikan *virtual environment* Python sudah aktif, lalu jalankan dari root direktori proyek (`diabetes_uts`):

```bash
python stage-3/8-compare-and-visualitation/compare-and-visualisation.py
```

---

## Output
Semua artefak visualisasi dan data disimpan di folder `output/` dan disalin ke `college/images/`:
1. `gambar-4-15-cm-baseline.png` — Confusion Matrix Baseline Decision Tree.
2. `gambar-4-16-cm-oversampling.png` — Confusion Matrix SMOTE Decision Tree.
3. `gambar-4-17-cm-undersampling.png` — Confusion Matrix RUS Decision Tree.
4. `gambar-cm-best-3-models.png` — Confusion Matrix 3 Model Terbaik (Decision Tree RUS, Naive Bayes RUS, SVM RUS) secara berdampingan.
5. `gambar-4-18-perbandingan-kinerja.png` — Gambar 4.18 multi-panel yang menyajikan perbandingan metrik evaluasi dan analisis kasus luput (FN) untuk ke-3 algoritma pada seluruh skenario.
6. `perbandingan-metrik.png` — Salinan grafik perbandingan metrik evaluasi.
7. `perbandingan-model-terbaik.png` — Head-to-Head perbandingan 4 metrik dan kesalahan klinis untuk 3 model terbaik (Skenario RUS).
8. `perbandingan-kesalahan-klinis.png` — Grafik perbandingan distribusi kesalahan diagnosis (FP vs FN) pada 9 konfigurasi model.
9. `perbandingan-radar.png` — Radar chart komparasi multi-metrik 3 algoritma terbaik.
10. `heatmap-ringkasan-metrik.png` — Heatmap visual seluruh nilai metrik pada 9 konfigurasi model.
11. `perbandingan-skenario.csv` / `perbandingan-semua-model.csv` — File CSV yang memuat nilai kuadran CM, metrik utama, spesifisitas, FPR, dan FNR seluruh model.

---

## Ringkasan Rekapitulasi Kinerja (9 Konfigurasi)

| Algoritma | Skenario | Akurasi | Presisi | Recall | F1-Score | FP (Alarm Palsu) | FN (Luput) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | Original | 77.18% | 75.76% | 49.02% | 59.52% | 8 psn | 26 psn |
| | SMOTE | 69.80% | 54.41% | 72.55% | 62.18% | 31 psn | 14 psn |
| | **RUS** | 74.50% | 59.42% | **80.39%** | **68.33%** | 28 psn | **10 psn** |
| **Naive Bayes** | Original | 72.48% | 60.87% | 54.90% | 57.73% | 18 psn | 23 psn |
| | SMOTE | 69.13% | 53.62% | 72.55% | 61.67% | 32 psn | 14 psn |
| | **RUS** | 72.48% | 57.81% | 72.55% | 64.35% | 27 psn | 14 psn |
| **SVM** | Original | 73.83% | 67.65% | 45.10% | 54.12% | 11 psn | 28 psn |
| | SMOTE | 75.17% | 62.50% | 68.63% | 65.42% | 21 psn | 16 psn |
| | **RUS** | **75.84%** | **61.90%** | 76.47% | **68.42%** | **24 psn** | 12 psn |

---

## Kesimpulan Komparatif Utama
1. **Resampling RUS Unggul Konsisten di Seluruh Algoritma:**  
   Ketiga model (Decision Tree, Naive Bayes, dan SVM) sama-sama mencapai performa terbaiknya (F1-Score dan Recall tertinggi) pada skenario **Random Under Sampling (RUS)**.
2. **Sensitivitas / Recall Tertinggi:**  
   **Decision Tree (RUS)** mencatatkan **Recall tertinggi (80.39%)** dengan kasus pasien luput paling minimal (**10 pasien**).
3. **Keseimbangan dan Akurasi Terbaik:**  
   **SVM (RUS)** mencatatkan **Akurasi tertinggi (75.84%)** dan **F1-Score tertinggi (68.42%)**, dengan kombinasi recall yang kuat (76.47%) dan angka alarm palsu yang lebih terkontrol (24 FP vs 28 FP pada DT).
