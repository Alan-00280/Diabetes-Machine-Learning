# Stage 3 - Compare and Visualisation (Komparasi 3 Algoritma)

## Pemilik
Raka (Anggota C)

## Deskripsi Tahap
Tahap 8 merupakan tahap puncak dari **Stage 3: Evaluasi dan Komparasi** yang membandingkan kinerja dari ketiga algoritma klasifikasi:
1. **Decision Tree** (dari `stage-3/7-1`)
2. **Gaussian Naive Bayes** (dari `stage-3/7-2`)
3. **Support Vector Machine / SVM** (dari `stage-3/7-3`)

Masing-masing model dievaluasi pada tiga skenario perlakuan data latih:
- **Original (Baseline)**: Data latih murni tanpa resampling (distribusi kelas timpang 66%:34%).
- **SMOTE (Oversampling)**: Penyeimbangan kelas sintetis minoritas (786 sampel seimbang 1:1).
- **RUS (Random Under-Sampling)**: Penyeimbangan kelas reduksi acak mayoritas (404 sampel seimbang 1:1).

Total 9 konfigurasi model dievaluasi secara adil pada **149 data uji independen** (98 Tidak Diabetes & 51 Diabetes).

---

## Input
- `stage-3/7-1-confusion-matrix-decision-tree/output/evaluation-metrics.csv`
- `stage-3/7-2-confusion-matrix-naive-bayes/output/evaluation-metrics.csv`
- `stage-3/7-3-confusion-matrix-svm/output/evaluation-metrics.csv`

---

## Cara Menjalankan
Pastikan virtual environment Python (`.venv`) aktif, lalu jalankan dari root direktori:

```bash
python stage-3/8-compare-and-visualitation/compare-and-visualisation.py
```

---

## Output
Semua artefak visualisasi dan data tersimpan di folder `output/`:
1. `perbandingan-skenario-lengkap.csv` — Data CSV memuat seluruh sel Confusion Matrix dan metrik lengkap dari 9 konfigurasi model.
2. `gambar-4-18-perbandingan-kinerja.png` — Multi-panel grafik perbandingan Akurasi, Presisi, Recall, dan F1-Score untuk seluruh model.
3. `perbandingan-kesalahan-klinis.png` — Diagram batang analisis kesalahan klinis: Alarm Palsu (FP) vs Luput (FN).
4. `perbandingan-radar.png` — Radar chart multidimensi membandingkan model terbaik tiap algoritma (skenario RUS).
5. `tabel-ranking-model.png` — Visualisasi tabel ranking model terbaik berdasarkan F1-Score dan Recall.

---

## Ringkasan Kinerja Lengkap (9 Model)

| No | Algoritma | Skenario | Akurasi | Presisi | Recall | F1-Score | FP (Alarm Palsu) | FN (Luput) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **SVM** | **RUS** | **75.84%** | **61.90%** | **76.47%** | **68.42%** | 24 pasien | **12 pasien** |
| 2 | Decision Tree | RUS | 74.50% | 59.42% | **80.39%** | 68.33% | 28 pasien | **10 pasien** |
| 3 | SVM | SMOTE | 75.17% | 62.50% | 68.63% | 65.42% | 21 pasien | 16 pasien |
| 4 | Naive Bayes | RUS | 72.48% | 57.81% | 72.55% | 64.35% | 27 pasien | 14 pasien |
| 5 | Decision Tree | SMOTE | 69.80% | 54.41% | 72.55% | 62.18% | 31 pasien | 14 pasien |
| 6 | Naive Bayes | SMOTE | 69.13% | 53.62% | 72.55% | 61.67% | 32 pasien | 14 pasien |
| 7 | Decision Tree | Original | 77.18% | 75.76% | 49.02% | 59.52% | 8 pasien | 26 pasien |
| 8 | Naive Bayes | Original | 72.48% | 60.87% | 54.90% | 57.73% | 18 pasien | 23 pasien |
| 9 | SVM | Original | 73.83% | 67.65% | 45.10% | 54.12% | 11 pasien | 28 pasien |

---

## Kesimpulan & Rekomendasi
1. **Dampak Resampling**: Semua model baseline (Original) menderita *False Negative* tinggi (23–28 pasien diabetes luput terdeteksi). Resampling (khususnya **RUS**) terbukti memangkas pasien luput hingga tersisa 10–12 pasien serta mendongkrak Recall di atas 76%.
2. **Model Terbaik Keseluruhan**: **SVM dengan Random Under-Sampling (SVM-RUS)** menjadi model dengan performa keseluruhan terbaik (**F1-Score 68.42%** dan **Akurasi 75.84%**), sehingga dideploy ke aplikasi web Streamlit (`app/`).
