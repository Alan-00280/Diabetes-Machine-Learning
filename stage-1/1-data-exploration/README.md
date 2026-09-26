# Stage 1 - Data Exploration

## Pemilik
Christianus Primavito

## Input
- `diabetes.csv` (root repository)

## Cara Menjalankan
Dijalankan dari root repository (folder `Diabetes-Machine-Learning`), dengan `.venv` sudah aktif:

```bash
python stage-1/1-data-exploration/data-exploration.py
```

## Output
Output berupa hasil print di terminal (bukan file), meliputi:
- Ukuran data (jumlah baris dan kolom)
- Info struktur data (tipe data tiap kolom, jumlah data non-null)
- Statistik dasar tiap kolom (mean, min, max, dll)
- Distribusi kelas target (Outcome)
- Persentase kelas target (Outcome)

Bukti output disimpan sebagai screenshot di folder `output/`.

## Catatan
- Dataset terdiri dari 2000 baris dan 9 kolom, seluruh kolom bertipe numerik (int64 dan float64), tidak ada data null (NaN).
- Distribusi kelas target tidak seimbang: Outcome 0 sebanyak 1316 data (65.8%), Outcome 1 sebanyak 684 data (34.2%).