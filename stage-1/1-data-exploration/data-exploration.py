import pandas as pd

# MENGAMBIL DATA
df = pd.read_csv("diabetes.csv")

# BARIS PROSES

# 1. Lihat ukuran data (jumlah baris dan kolom)
print("=== Ukuran Data (baris, kolom) ===")
print(df.shape)

# 2. Lihat tipe data tiap kolom
print("\n=== Info Struktur Data ===")
print(df.info())

# 3. Lihat statistik dasar tiap kolom (rata-rata, min, max, dll)
print("\n=== Statistik Dasar ===")
print(df.describe())

# 4. Lihat distribusi kelas target (Outcome)
print("\n=== Distribusi Kelas Target (Outcome) ===")
print(df["Outcome"].value_counts())

print("\n=== Persentase Kelas Target (Outcome) ===")
print(df["Outcome"].value_counts(normalize=True) * 100)

# OUTPUT
# Jalankan file ini, hasil tampil di terminal