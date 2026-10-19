# Stage 3 - Confusion Matrix Support Vector Machine (SVM)

## Pemilik
Luthfi Alan Perdana (Anggota A - 434241052)

## Input
- `stage-2/6-3-svm-and-testing/output/test-predictions.csv` — 149 baris data uji, berisi kolom kunci jawaban `Aktual (y_test)` dan hasil prediksi ketiga model SVM (`Pred_Original`, `Pred_SMOTE`, `Pred_RUS`).
- `stage-2/6-3-svm-and-testing/output/model-metrics.csv` — digunakan untuk verifikasi dan validasi silang (*cross-validation check*) dengan hasil tahap 6.3.

## Metode
Setiap model SVM dibandingkan hasil prediksinya dengan ground truth/kunci jawaban (`y_test`), kemudian dihitung Confusion Matrix 2x2.
Empat kuadran sel Confusion Matrix dianalisis dalam konteks medis diagnosis diabetes:

- **Benar Sehat (True Negative / TN)**: Pasien kenyataannya tidak diabetes, dan diprediksi tidak diabetes.
- **Alarm Palsu (False Positive / FP)**: Pasien kenyataannya tidak diabetes, tetapi diprediksi diabetes.
- **Luput (False Negative / FN)**: Pasien kenyataannya diabetes, tetapi diprediksi tidak diabetes (berbahaya secara klinis / berisiko fatal tanpa penanganan).
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
python stage-3/7-3-confusion-matrix-svm/confusion-matrix.py
```

## Output
- `output/evaluation-metrics.csv` — Confusion matrix dan 4 metrik evaluasi untuk ketiga skenario SVM.
- `output/confusion-matrix-persen.png` — Heatmap confusion matrix ketiga skenario dalam persentase per baris aktual.
- `output/perbandingan-metrik.png` — Diagram batang komparasi 4 metrik evaluasi ketiga skenario.
- `output/perbandingan-kesalahan.png` — Diagram batang analisis kesalahan klasifikasi klinis (Alarm Palsu vs Luput).

## Hasil Evaluasi Ringkas

| Skenario | TN (Benar Sehat) | FP (Alarm Palsu) | FN (Luput) | TP (Benar Diabetes) | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|---|---|
| **Original** | 87 | 11 | 28 | 23 | 73.83% | 67.65% | 45.10% | 54.12% |
| **SMOTE** | 77 | 21 | 16 | 35 | 75.17% | 62.50% | 68.63% | 65.42% |
| **RUS** | 74 | 24 | 12 | 39 | 75.84% | 61.90% | 76.47% | 68.42% |

## Catatan Analisis Medis
- Normalisasi persentase per baris dilakukan karena data uji memiliki distribusi kelas imbalanced (98 Sehat vs 51 Diabetes).
- Model baseline (**SVM-Original**) memiliki sensitivitas/Recall yang sangat rendah (45.10%), di mana **28 pasien diabetes luput terdiagnosis**. Hal ini sangat berbahaya dalam skenario skrining klinis.
- Penerapan teknik resampling (**SMOTE** dan **RUS**) berhasil mendongkrak Recall secara dramatis:
  - **SMOTE** menurunkan angka luput menjadi 16 pasien (Recall 68.63%, F1-Score 65.42%).
  - **RUS** menekan angka luput menjadi hanya 12 pasien (Recall 76.47%, F1-Score 68.42%).
- **Skenario RUS (Random Under Sampling)** merupakan model terbaik untuk SVM pada dataset ini, menghasilkan **Akurasi tertinggi (75.84%)**, **Recall tertinggi (76.47%)**, dan **F1-Score tertinggi (68.42%)**.
