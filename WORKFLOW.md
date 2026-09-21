# Panduan Workflow Proyek

Dokumen ini menjadi aturan kerja bersama untuk proyek klasifikasi data diabetes. Semua anggota wajib mengikuti urutan tahap, menyimpan kode dan output pada tahap yang sama, serta menggunakan branch pribadi.

## 1. Prasyarat

- Python 3.10 atau lebih baru
- Git
- Akses ke repository remote kelompok
- VS Code atau editor Python lain

Semua perintah berikut dijalankan dari folder root repository, yaitu folder yang berisi `diabetes.csv`.

## 2. Mempersiapkan Environment

### 2.1 Membuat dan mengaktifkan virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Jika berhasil, nama environment biasanya muncul di awal prompt terminal sebagai `(.venv)`. Setiap anggota membuat `.venv` sendiri di komputer masing-masing. Folder `.venv` tidak boleh di-commit.

### 2.2 Install library

Setelah environment aktif, jalankan:

```bash
pip install -r requirements.txt
```

Cek instalasi:

```bash
python -m pip list
```

Catatan: gunakan hanya import yang benar-benar diperlukan oleh file tersebut. Setelah environment stabil, sebaiknya tim menambahkan `requirements.txt` agar instalasi semua anggota dapat direproduksi dengan `python -m pip install -r requirements.txt`.

## 3. Aturan Struktur Setiap Folder

Setiap folder tugas harus menyimpan kode dan hasilnya secara berdampingan. Gunakan pola berikut:

```text
nama-folder/
|-- README.md                 # tujuan, input, cara menjalankan, dan handoff
|-- process.py                # script utama tahap tersebut
`-- output/
    |-- cleaned-data.csv      # contoh output data
    |-- result.json            # contoh output metrik/parameter
    `-- figure.png             # contoh output visualisasi
```

Nama file boleh disesuaikan kebutuhan tahap, tetapi aturan berikut wajib dipenuhi:

1. Kode harus dapat dijalankan ulang dari root repository menggunakan environment `.venv`.
2. Kode tidak boleh bergantung pada file lokal di luar repository.
3. Output yang dibutuhkan tahap berikutnya harus disimpan di folder `output/` dan di-commit ke Git.
4. Jangan menyimpan `.venv/`, cache Python, atau file sementara ke repository.
5. `README.md` di dalam folder harus mencantumkan input, perintah menjalankan, daftar output, dan pemilik pekerjaan.
6. Gunakan random seed yang konsisten jika ada split data atau resampling, sehingga output dapat direproduksi.

## 4. Urutan Tahap dan Handoff Output

Alur utama:

```text
diabetes.csv
  -> stage-1/data-exploration
  -> stage-1/missing-and-duplication
  -> stage-1/outlier-and-split-data
  -> stage-2/data-transformation
  -> stage-2/resampling
  -> stage-2/decision-tree-and-testing
  -> stage-3/compare-and-visualitation
  -> stage-3/confusion-matrix
  -> final-app
```

### Stage 1: Persiapan dan pembagian data

| Folder | Tugas | Input | Handoff wajib |
|---|---|---|---|
| `stage-1/data-exploration` | Memahami ukuran data, tipe kolom, distribusi target, dan statistik awal | `diabetes.csv` | Ringkasan eksplorasi dan data awal yang dipakai tahap berikutnya |
| `stage-1/missing-and-duplication` | Mendeteksi dan menangani missing value serta data duplikat | Output eksplorasi | Dataset setelah missing value dan duplikasi ditangani |
| `stage-1/outlier-and-split-data` | Menangani outlier lalu membagi data menjadi train dan test, target 80:20 | Dataset hasil pembersihan | `X_train`, `X_test`, `y_train`, dan `y_test` |

### Stage 2: Transformasi, resampling, dan model

| Folder | Tugas | Input | Handoff wajib |
|---|---|---|---|
| `stage-2/data-transformation` | Scaling fitur setelah split untuk mencegah data leakage | Data train/test dari Stage 1 | Data train/test hasil transformasi dan scaler jika diperlukan |
| `stage-2/resampling` | Oversampling dan undersampling hanya pada data training | Data train hasil transformasi | Dataset train hasil masing-masing metode resampling |
| `stage-2/decision-tree-and-testing` | Melatih Decision Tree dan melakukan prediksi pada data test | Data train hasil resampling dan data test | Model/prediksi serta metrik pengujian awal |

### Stage 3: Evaluasi dan visualisasi

| Folder | Tugas | Input | Handoff wajib |
|---|---|---|---|
| `stage-3/compare-and-visualitation` | Membandingkan hasil model tanpa resampling, oversampling, dan undersampling | Metrik dari Stage 2 | Tabel perbandingan dan grafik evaluasi |
| `stage-3/confusion-matrix` | Membuat confusion matrix dan evaluasi detail | Label aktual dan prediksi | Gambar confusion matrix serta accuracy, precision, recall, dan F1-score |

### Final app

`final-app/` digunakan untuk integrasi atau demonstrasi akhir setelah pipeline dan hasil evaluasi disepakati. Folder ini saat ini belum berisi implementasi. Jangan membangun final app dari output lokal yang belum di-commit.

## 5. Prosedur Git untuk Setiap Anggota

### 5.1 Clone dan konfigurasi awal

```bash
git clone <URL-REPOSITORY>
cd diabetes_uts
git config user.name "Nama Lengkap"
git config user.email "email@example.com"
```

### 5.2 Membuat branch pribadi

Setiap anggota wajib memiliki branch sesuai namanya sendiri. Gunakan huruf kecil dan tanda hubung, misalnya:

```bash
git switch main
git pull origin main
git switch -c nama-lengkap
```

Contoh:

```bash
git switch -c luthfi
```

Jangan mengerjakan perubahan langsung di `main`, dan jangan menggunakan branch bersama untuk pekerjaan pribadi.

### 5.3 Sebelum mulai bekerja: pull terbaru

```bash
git switch nama-lengkap
git pull --rebase origin main
```

Jika ada konflik, selesaikan konflik pada file terkait, lalu jalankan:

```bash
git add <file-yang-sudah-diperbaiki>
git rebase --continue
```

### 5.4 Menyimpan pekerjaan dan push

Sebelum commit, pastikan script dapat dijalankan ulang dan output sudah dibuat dari script tersebut.

```bash
git status
git add stage-1/nama-folder/
git commit -m "feat: tambah proses nama tahap"
git push -u origin nama-lengkap
```

Gunakan commit kecil dan jelas. Sertakan kode, `README.md` tahap, dan output yang menjadi handoff. Jangan melakukan `git add .` secara membabi buta karena dapat memasukkan `.venv` atau file sementara.

### 5.5 Mengambil hasil anggota lain

Setelah anggota sebelumnya menyelesaikan tahap dan menggabungkannya ke `main`, anggota berikutnya wajib memperbarui branch-nya:

```bash
git switch nama-lengkap
git pull --rebase origin main
```

Kemudian gunakan file di `output/` tahap sebelumnya sebagai input. Jangan menyalin output secara manual ke folder pribadi; gunakan path repository dan dokumentasikan nama file inputnya.

## 6. Aturan Handoff Antar-Anggota

Setiap handoff harus memenuhi checklist berikut:

- [ ] Script/notebook tahap sudah tersimpan di folder tahap yang sesuai.
- [ ] Output sudah tersimpan di `output/` dan dapat dibuka oleh anggota berikutnya.
- [ ] README tahap menjelaskan input, cara menjalankan, output, dan asumsi.
- [ ] Nama file input dan output dicatat secara eksplisit.
- [ ] Tidak ada path absolut seperti `C:\Users\...` di dalam kode.
- [ ] Perubahan sudah diuji dengan `.venv` dan tidak memasukkan `.venv/` ke commit.
- [ ] Commit sudah di-push ke branch pribadi dan diajukan melalui Pull Request.

Alur Pull Request: push branch pribadi -> buka Pull Request ke `main` -> minta anggota lain melakukan review -> perbaiki jika diperlukan -> merge -> anggota tahap berikutnya melakukan pull dari `main`.

## 7. Checklist Sebelum Pengumpulan

- Semua folder tahap memiliki kode, output, dan README tahap.
- Semua output dapat ditelusuri dari script pembuatnya.
- Tidak ada data train yang terkena resampling sebelum split.
- Scaling dilakukan setelah split dan scaler tidak di-fit pada data test.
- Hasil tiga kondisi model dapat dibandingkan secara konsisten.
- Confusion matrix dan metrik evaluasi tersedia.
- `final-app/` hanya menggunakan output final yang sudah disepakati.
- Branch pribadi sudah digabungkan ke `main` melalui Pull Request.
