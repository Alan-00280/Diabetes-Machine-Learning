# Stage 3 - Confusion Matrix Naive Bayes

## Pemilik
Raka (Anggota C)

## Input
- `stage-2/6-2-naive-bayes-and-testing/output/test-predictions.csv` — 149 baris data uji, berisi kolom kunci jawaban `Aktual (y_test)` dan hasil prediksi ketiga model Naive Bayes (`Pred_Original`, `Pred_SMOTE`, `Pred_RUS`).
- `stage-2/6-2-naive-bayes-and-testing/output/model-metrics.csv` — digunakan untuk verifikasi dan validasi silang (*cross-validation check*) dengan hasil tahap 6.

## Metode
Setiap model Naive Bayes dibandingkan hasil prediksinya dengan ground truth/kunci jawaban (`y_test`), kemudian dihitung Confusion Matrix 2x2.
Empat kuadran sel Confusion Matrix dianalisis dalam konteks medis diagnosis diabetes:

- **Benar Sehat (True Negative / TN)**: Pasien kenyataannya tidak diabetes, dan diprediksi tidak diabetes.
- **Alarm Palsu (False Positive / FP)**: Pasien kenyataannya tidak diabetes, tetapi diprediksi diabetes.
- **Luput (False Negative / FN)**: Pasien kenyataannya diabetes, tetapi diprediksi tidak diabetes (berbahaya secara klinis).
- **Benar Diabetes (True Positive / TP)**: Pasien kenyataannya diabetes, dan diprediksi diabetes.

Dari keempat sel diturunkan metrik evaluasi formal:

```text
Accuracy  = (Benar Sehat + Benar Diabetes) / Jumlah Seluruh Data Uji
Precision = Benar Diabetes / (Benar Diabetes + Alarm Palsu)
Recall    = Benar Diabetes / (Benar Diabetes + Luput)
F1-Score  = 2 x (Precision x Recall) / (Precision + Recall)
```

## Cara Menjalankan
Dijalankan dari root repository (`Diabetes-Machine-Learning`) dengan virtual environment `.venv` yang aktif:

```bash
python stage-3/7-2-confusion-matrix-naive-bayes/confusion-matrix.py
```

## Output
- `output/evaluation-metrics.csv` — Confusion matrix dan 4 metrik evaluasi untuk ketiga skenario Naive Bayes.
- `output/confusion-matrix-persen.png` — Heatmap confusion matrix ketiga skenario dalam persentase per baris aktual.
- `output/perbandingan-metrik.png` — Diagram batang komparasi 4 metrik evaluasi ketiga skenario.
- `output/perbandingan-kesalahan.png` — Diagram batang analisis kesalahan klasifikasi (Alarm Palsu vs Luput).

## Hasil Evaluasi Ringkas

| Skenario | TN (Benar Sehat) | FP (Alarm Palsu) | FN (Luput) | TP (Benar Diabetes) | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|---|---|
| **Original** | 80 | 18 | 23 | 28 | 72.48% | 60.87% | 54.90% | 57.73% |
| **SMOTE** | 66 | 32 | 14 | 37 | 69.13% | 53.62% | 72.55% | 61.67% |
| **RUS** | 71 | 27 | 14 | 37 | 72.48% | 57.81% | 72.55% | 64.35% |

## Catatan Analisis Medis
- Normalisasi persentase per baris dilakukan karena data uji memiliki distribusi kelas imbalanced (98 Sehat vs 51 Diabetes).
- Teknik resampling (baik SMOTE maupun RUS) berhasil memangkas pasien diabetes yang **Luput (FN)** dari 23 orang menjadi hanya 14 orang (peningkatan Recall dari 54.90% menjadi 72.55%).
- **Skenario RUS** menghasilkan performa Naive Bayes terbaik secara seimbang dengan Akurasi 72.48% dan F1-Score tertinggi (64.35%).
