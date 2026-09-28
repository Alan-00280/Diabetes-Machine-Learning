# Stage 2 - Data Transformation

## Pemilik
Christianus Primavito

## Input
- `stage-1/3-outlier-and-split-data/output/train-diabetes.csv` (595 baris x 9 kolom)
- `stage-1/3-outlier-and-split-data/output/test-diabetes.csv` (149 baris x 9 kolom)

## Metode
Z-Score Standardization dengan rumus:

```text
z = (x - mean) / std
```

Mean dan std dihitung dari data train saja, lalu nilai tersebut dipakai untuk
mentransformasi data train dan data test. Hasilnya data train memiliki mean 0 dan std 1.

## Cara Menjalankan
Dijalankan dari root repository (folder `Diabetes-Machine-Learning`), dengan `.venv` sudah aktif:

```bash
python stage-2/4-data-transformation/data-transformation.py
```

## Output
- `output/train-transformed.csv` - data train hasil transformasi (8 fitur + kolom `Outcome`)
- `output/test-transformed.csv` - data test hasil transformasi (8 fitur + kolom `Outcome`)
- `output/scaling-comparison.png` - grafik boxplot skala fitur sebelum dan sesudah transformasi

Bukti output di terminal disimpan sebagai screenshot di folder `output/`.

## Handoff untuk Tahap Berikutnya
- `stage-2/5-resampling` memakai **`train-transformed.csv`** sebagai input, lalu melakukan
  oversampling dan undersampling **hanya pada data train**.
- `stage-2/6-decision-tree-and-testing` melatih model dari data train hasil resampling dan
  menguji model memakai **`test-transformed.csv`**.
- Kedua file mempertahankan nama dan urutan kolom yang sama seperti input, sehingga kolom
  fitur dan kolom target `Outcome` dapat dipisah dengan cara yang sama.
- Data test tidak boleh ikut di-resampling agar hasil evaluasi tetap menggambarkan kondisi
  data yang tidak seimbang.

## Asumsi dan Catatan
- Mean dan std dihitung **hanya dari data train**, kemudian dipakai untuk mentransformasi train
  dan test. Ini mencegah data leakage karena nilai pada data test tidak ikut menentukan mean
  dan std.
- Kolom target `Outcome` tidak ditransformasi, nilainya tetap 0 dan 1.
- Seluruh fitur bertipe numerik, sehingga tidak diperlukan encoding.
- Decision Tree tidak sensitif terhadap skala fitur, namun transformasi tetap dilakukan karena
  tahap resampling berikutnya bekerja berbasis jarak dan sebagai praktik standar pemodelan.
- Transformasi bersifat deterministik, tidak memakai random seed.
