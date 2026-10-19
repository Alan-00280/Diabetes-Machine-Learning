# Stage 3 - Confusion Matrix SVM

## Pemilik
Christianus Primavito

## Input
- `stage-2/6-3-svm-and-testing/output/test-predictions.csv` — 149 baris data uji, berisi kolom
  kunci jawaban `Aktual (y_test)` dan prediksi ketiga skenario (`Pred_Original`, `Pred_SMOTE`, `Pred_RUS`)
- `stage-2/6-3-svm-and-testing/output/model-metrics.csv` — dipakai hanya untuk pengecekan silang
  hasil tahap ini dengan hasil pengujian tahap 6-3

## Metode
Setiap prediksi dibandingkan dengan kunci jawaban, lalu dihitung confusion matrix-nya.
Empat sel confusion matrix diberi nama:

- **Benar Sehat** (True Negative) — kenyataan tidak diabetes, diprediksi tidak diabetes
- **Alarm Palsu** (False Positive) — kenyataan tidak diabetes, diprediksi diabetes
- **Luput** (False Negative) — kenyataan diabetes, diprediksi tidak diabetes
- **Benar Diabetes** (True Positive) — kenyataan diabetes, diprediksi diabetes

Dari keempat sel tersebut diturunkan empat metrik:

```text
Accuracy  = (Benar Sehat + Benar Diabetes) / jumlah seluruh data uji
Precision = Benar Diabetes / (Benar Diabetes + Alarm Palsu)
Recall    = Benar Diabetes / (Benar Diabetes + Luput)
F1-Score  = 2 x (Precision x Recall) / (Precision + Recall)
```

## Cara Menjalankan
Dijalankan dari root repository (folder `Diabetes-Machine-Learning`), dengan `.venv` sudah aktif:

```bash
python stage-3/7-3-confusion-matrix-svm/confusion-matrix.py
```

## Output
- `output/evaluation-metrics.csv` — confusion matrix dan empat metrik untuk ketiga skenario
- `output/confusion-matrix-persen.png` — confusion matrix ketiga skenario dalam persen per baris
- `output/perbandingan-metrik.png` — grafik batang empat metrik pada ketiga skenario
- `output/perbandingan-kesalahan.png` — grafik batang jumlah alarm palsu dan luput tiap skenario

Bukti output di terminal disimpan sebagai screenshot di folder `output/`.

## Hasil

| Skenario | Benar Sehat | Alarm Palsu | Luput | Benar Diabetes | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|---|---|
| Original | 87 | 11 | 28 | 23 | 0.7383 | 0.6765 | 0.4510 | 0.5412 |
| SMOTE | 77 | 21 | 16 | 35 | 0.7517 | 0.6250 | 0.6863 | 0.6542 |
| RUS | 74 | 24 | 12 | 39 | 0.7584 | 0.6190 | 0.7647 | 0.6842 |

Pola yang muncul sama seperti Decision Tree dan Naive Bayes: RUS memberi keseimbangan terbaik,
sedangkan skenario Original paling jarang salah tuduh tetapi paling banyak meluputkan pasien
diabetes (28 dari 51).

## Catatan
- Confusion matrix ditampilkan dalam **persen per baris** karena jumlah kelas pada data uji tidak
  seimbang (98 tidak diabetes dan 51 diabetes). Angka mentah membuat kelas yang jumlahnya sedikit
  terlihat remeh, sedangkan versi persen memperlihatkan proporsi kesalahan tiap kelas secara adil.
- `Precision` dan `Recall` dihitung untuk kelas **1 (diabetes)** sebagai kelas positif.
- Angka pada tahap ini dihitung dari prediksi tahap 6-3 yang sama, sehingga hasilnya sama dengan
  `model-metrics.csv`. Pencocokan tersebut dicetak di akhir script sebagai pengecekan.
- Menjalankan ulang pelatihan SVM (`SVC` kernel rbf + `CalibratedClassifierCV`) menghasilkan
  prediksi yang identik, sehingga angka pada tahap ini dapat direproduksi dari input tahap 6-3.
- Model SVM dilatih memakai data yang sudah ditransformasi pada tahap 4.

## Handoff untuk Tahap Berikutnya
- `stage-3/8-compare-and-visualitation` dapat memakai `evaluation-metrics.csv` beserta grafik pada
  folder `output/` untuk membandingkan kinerja Decision Tree, Naive Bayes, dan SVM.
- Angka pada `evaluation-metrics.csv` menjadi dasar penulisan bagian evaluasi SVM pada laporan.
