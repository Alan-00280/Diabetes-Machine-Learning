# Stage 3 - Confusion Matrix

## Pemilik
Christianus Primavito

## Input
- `stage-2/6-decision-tree-and-testing/output/test-predictions.csv` — 149 baris data uji, berisi
  kolom kunci jawaban `Aktual (y_test)` dan prediksi ketiga model (`Pred_Original`, `Pred_SMOTE`, `Pred_RUS`)
- `stage-2/6-decision-tree-and-testing/output/model-metrics.csv` — dipakai hanya untuk pengecokan
  silang hasil tahap ini dengan hasil pengujian tahap 6

## Metode
Setiap model dibandingkan prediksinya dengan kunci jawaban, lalu dihitung confusion matrix-nya.
Empat sel confusion matrix diberi nama:

- **Benar Sehat** — kenyataan tidak diabetes, diprediksi tidak diabetes
- **Alarm Palsu** — kenyataan tidak diabetes, diprediksi diabetes
- **Luput** — kenyataan diabetes, diprediksi tidak diabetes
- **Benar Diabetes** — kenyataan diabetes, diprediksi diabetes

Dari keempat sel tersebut diturunkan empat metrik dengan rumus:

```text
Accuracy  = (Benar Sehat + Benar Diabetes) / jumlah seluruh data uji
Precision = Benar Diabetes / (Benar Diabetes + Alarm Palsu)
Recall    = Benar Diabetes / (Benar Diabetes + Luput)
F1-Score  = 2 x (Precision x Recall) / (Precision + Recall)
```

## Cara Menjalankan
Dijalankan dari root repository (folder `Diabetes-Machine-Learning`), dengan `.venv` sudah aktif:

```bash
python stage-3/7-confusion-matrix/confusion-matrix.py
```

## Output
- `output/evaluation-metrics.csv` — confusion matrix dan empat metrik untuk ketiga skenario
- `output/confusion-matrix-persen.png` — confusion matrix ketiga skenario dalam persen per baris
- `output/perbandingan-metrik.png` — grafik batang empat metrik pada ketiga skenario
- `output/perbandingan-kesalahan.png` — grafik batang jumlah alarm palsu dan luput tiap skenario

Bukti output di terminal disimpan sebagai screenshot di folder `output/`.

## Catatan
- Confusion matrix ditampilkan dalam **persen per baris** karena jumlah kelas pada data uji tidak
  seimbang (98 tidak diabetes dan 51 diabetes). Angka mentah membuat kelas yang jumlahnya sedikit
  terlihat remeh, sedangkan versi persen memperlihatkan proporsi kesalahan pada masing-masing kelas
  secara adil.
- `Precision` dan `Recall` dihitung untuk kelas **1 (diabetes)** sebagai kelas positif.
- Hasil tahap ini dihitung dari prediksi tahap 6 yang sama, sehingga angkanya harus sama dengan
  `model-metrics.csv`. Pencocokan tersebut dicetak di akhir script sebagai pengecekan.

## Handoff untuk Tahap Berikutnya
- `stage-3/8-compare-and-visualitation` memakai `evaluation-metrics.csv` beserta grafik pada folder
  `output/` sebagai bahan perbandingan antar skenario.
- Angka pada `evaluation-metrics.csv` juga menjadi dasar penulisan BAB IV bagian evaluasi.
