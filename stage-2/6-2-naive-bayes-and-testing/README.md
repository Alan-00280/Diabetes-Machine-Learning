# Stage 2 - Naive Bayes and Testing

## Pemilik
Christianus Primavito

## Input
- `stage-2/4-data-transformation/output/test-transformed.csv` — data uji (149 baris, 8 fitur + `Outcome`)
- `stage-2/4-data-transformation/output/train-transformed.csv` — data train tanpa resampling
- `stage-2/5-resampling/output/train-oversampling-smote.csv` — data train hasil oversampling (SMOTE)
- `stage-2/5-resampling/output/train-undersampling-rus.csv` — data train hasil undersampling (RUS)

Data uji selalu memakai `test-transformed.csv` yang sama seperti tahap 6-1, sehingga hasil ketiga
skenario dapat dibandingkan secara adil.

## Metode
Klasifikasi memakai `sklearn.naive_bayes.GaussianNB`, sesuai modul Pertemuan 8 - Klasifikasi.
Varian Gaussian dipilih karena seluruh fitur bertipe angka kontinu dan sudah ditransformasi
pada tahap 4.

Tiga skenario data latih:

| Skenario | Data latih | Jumlah baris | Komposisi kelas |
|---|---|---|---|
| Original | `train-transformed.csv` | 595 | 393 : 202 |
| SMOTE | `train-oversampling-smote.csv` | 786 | 393 : 393 |
| RUS | `train-undersampling-rus.csv` | 404 | 202 : 202 |

Metrik yang dihitung: accuracy, precision, recall (sensitivity), F1-score, serta specificity
sebagai pelengkap. Precision, recall, dan F1 dihitung untuk kelas 1 (Diabetes) sebagai kelas positif.

## Cara Menjalankan
Dijalankan dari root repository (folder `Diabetes-Machine-Learning`), dengan `.venv` sudah aktif:

```bash
python stage-2/6-2-naive-bayes-and-testing/naive-bayes-and-testing.py
```

## Output
- `output/model-metrics.csv` — metrik ketiga skenario
- `output/test-predictions.csv` — kunci jawaban dan prediksi ketiga skenario pada data uji
- `output/ss1-metrics-table.png` — tabel perbandingan metrik
- `output/ss2-confusion-matrix.png` — confusion matrix ketiga skenario
- `output/ss3-sebaran-probabilitas.png` — sebaran probabilitas prediksi diabetes

Bukti output di terminal disimpan sebagai screenshot di folder `output/`.

## Handoff untuk Tahap Berikutnya
- `stage-3/7-2-confusion-matrix-naive-bayes` memakai **`test-predictions.csv`** sebagai input.
  Nama kolomnya sengaja dibuat sama dengan keluaran tahap 6-1
  (`Aktual (y_test)`, `Pred_Original`, `Pred_SMOTE`, `Pred_RUS`), sehingga script tahap 7-1 dapat
  dipakai ulang dengan hanya mengubah folder inputnya.
- `stage-3/8-compare-and-visualitation` dapat memakai `model-metrics.csv` untuk membandingkan
  performa Naive Bayes dengan Decision Tree dan SVM.

## Asumsi dan Catatan
- `GaussianNB` bersifat deterministik sehingga tidak memakai random seed, berbeda dengan
  Decision Tree yang memakai `random_state`.
- Sensitivitas sama nilainya dengan recall kelas diabetes, yaitu kemampuan model menemukan
  pasien yang benar-benar diabetes. Specificity adalah kemampuan model mengenali pasien sehat.
- Model dilatih memakai data yang sudah ditransformasi pada tahap 4. Mean dan standar deviasi
  untuk transformasi data baru harus memakai parameter dari `stage-1/3-outlier-and-split-data`
  (data train sebelum scaling), bukan dari `diabetes.csv` mentah.
- Hasil pada data uji menunjukkan pola yang sama seperti Decision Tree: RUS memberi keseimbangan
  terbaik antara menemukan pasien diabetes dan menghindari salah tuduh, sedangkan Original paling
  jarang salah tuduh tetapi paling banyak meluputkan pasien diabetes.
