# Workflow Singkat

## 1. Setup Project

```powershell
git clone <URL-REPOSITORY>
cd diabetes_uts
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Buat Branch Pribadi

Jangan bekerja langsung di `main`.

```bash
git switch main
git pull origin main
git switch -c nama-anggota
```

Contoh:

```bash
git switch -c luthfi
```

## 3. Sebelum Bekerja

```bash
git switch nama-anggota
git pull --rebase origin main
```

Lalu:

- Kerjakan folder yang ditugaskan.
- Ambil output dari pekerjaan sebelumnya.
- Simpan kode di folder tugas.
- Simpan hasil di folder `output/`.
- Perbarui `README.md` jika diperlukan.

## 4. Commit dan Push

```bash
git status
git add <folder-yang-dikerjakan>
git commit -m "feat: tambah proses dan output"
git push -u origin nama-anggota
```

Pastikan sebelum push:

- [ ] Kode dapat dijalankan.
- [ ] Output sudah tersedia.
- [ ] Tidak ada `.venv/`, cache, atau file sementara.
- [ ] Path yang digunakan bukan path absolut komputer pribadi.

## 5. Berbagi Hasil

1. Push branch pribadi.
2. Buat Pull Request ke `main`.
3. Minta anggota lain melakukan review.
4. Perbaiki jika ada masukan.
5. Merge setelah disetujui.
6. Beri tahu anggota berikutnya.

Anggota berikutnya menjalankan:

```bash
git switch nama-anggota
git pull --rebase origin main
```

## 6. Checklist Akhir

- [ ] Kode dan output sudah tersimpan di folder yang sesuai.
- [ ] Hasil sudah dapat digunakan anggota berikutnya.
- [ ] Branch pribadi sudah di-push.
- [ ] Pull Request sudah di-merge ke `main`.
- [ ] `.venv/` tidak masuk repository.
