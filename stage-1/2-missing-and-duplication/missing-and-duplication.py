# Tolong ganti nama file template.py sesuaikan dengan bagianmu

from pathlib import Path

import numpy as np
import pandas as pd

# MENGAMBIL DATA
# Bisa pakai data output tahap sebelumnya
project_root = Path(__file__).resolve().parents[2]
input_path = project_root / "diabetes.csv"
output_dir = Path(__file__).resolve().parent / "output"
output_path = output_dir / "cleaned-diabetes.csv"

df = pd.read_csv(input_path)

# BARIS PROSES
print("=== KONDISI DATA SEBELUM PEMBERSIHAN ===")
print(f"Ukuran data: {df.shape[0]} baris, {df.shape[1]} kolom")
print("Missing value NaN per kolom:")
print(df.isna().sum().to_string())

medical_columns = [
	"Glucose",
	"BloodPressure",
	"SkinThickness",
	"Insulin",
	"BMI",
]

# Nol pada kolom medis berikut merupakan missing value terselubung.
zero_missing = df[medical_columns].eq(0).sum()
print("\nMissing value terselubung (nilai 0) per kolom:")
print(zero_missing.to_string())

duplicate_count_before = int(df.duplicated().sum())
print(f"\nJumlah baris duplikat sebelum pembersihan: {duplicate_count_before}")
if duplicate_count_before:
	print("Contoh baris duplikat:")
	print(df[df.duplicated(keep=False)].head(3).to_string(index=False))

# Ubah sentinel 0 menjadi NaN, lalu isi dengan median kolom agar tidak
# menghapus terlalu banyak data.
df[medical_columns] = df[medical_columns].replace(0, np.nan)
for column in medical_columns:
	df[column] = df[column].fillna(df[column].median())

df = df.drop_duplicates().reset_index(drop=True)

# OUTPUT
output_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("\n=== KONDISI DATA SETELAH PEMBERSIHAN ===")
print(f"Ukuran data: {df.shape[0]} baris, {df.shape[1]} kolom")
print("Missing value NaN per kolom:")
print(df.isna().sum().to_string())
print(f"Jumlah baris duplikat setelah pembersihan: {int(df.duplicated().sum())}")
print("Contoh data bersih:")
print(df.head().to_string(index=False))
print(f"\nFile output: {output_path}")
