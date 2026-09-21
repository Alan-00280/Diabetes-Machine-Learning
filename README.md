# Klasifikasi Data Diabetes Imbalanced dengan Decision Tree

Proyek ini merupakan implementasi alur kerja machine learning untuk mengklasifikasikan
data diabetes yang bersifat imbalanced menggunakan algoritma Decision Tree, dengan
penerapan teknik resampling (oversampling dan undersampling) pada data latih.

Dikerjakan sebagai tugas Ujian Tengah Semester mata kuliah Praktikum Pembelajaran Mesin,
Program Studi D4 Teknik Informatika, Fakultas Vokasi, Universitas Airlangga.

## Alur Kerja

Input → Preprocessing → Split Data → Transformation → Resampling (data training) → Training → Testing → Evaluasi

1. **Preprocessing** — deteksi dan penanganan missing value, duplikasi data, dan outlier
2. **Split Data** — 80% data training, 20% data testing
3. **Transformation** — scaling fitur (dilakukan setelah split untuk menghindari data leakage)
4. **Resampling** — oversampling dan undersampling, hanya diterapkan pada data training
5. **Training** — model Decision Tree
6. **Evaluasi** — confusion matrix, akurasi, presisi, recall, F1-Score

## Struktur Repo

```text
diabetes_uts/
|-- diabetes.csv
|-- WORKFLOW.md
|-- stage-1/
|   |-- data-exploration/
|   |-- missing-and-duplication/
|   `-- outlier-and-split-data/
|-- stage-2/
|   |-- data-transformation/
|   |-- resampling/
|   `-- decision-tree-and-testing/
|-- stage-3/
|   |-- compare-and-visualitation/
|   `-- confusion-matrix/
`-- final-app/
```

Setiap folder tahap menyimpan kode Python/notebook dan outputnya. Output dari satu folder menjadi input untuk folder berikutnya dan harus di-commit agar dapat digunakan anggota lain. Detail struktur file, setup `.venv`, import library, aturan handoff, serta prosedur branch, pull, commit, push, dan Pull Request tersedia di [WORKFLOW.md](WORKFLOW.md).

Semua anggota wajib membuat branch pribadi sesuai nama masing-masing, misalnya `luthfi-alan-perdana`. Jangan melakukan pekerjaan langsung pada branch `main`.

## Tools

Python, pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn

## Anggota Kelompok

| Nama | NIM |
|---|---|
| Luthfi Alan Perdana | 434241052 |
| Muhammad Raka Razzani  | 434241056 |
| Christianus Primavito | 434241058 |

## Dosen Pengampu

- Dr. Indah Werdiningsih, S.Si., M.Kom
- Barry Nuqoba, S.Si., M.Kom., Ph.D
- Purbandini, S.Si., M.Kom