from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Agar tabel tidak terpotong saat di-print
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

# 1. MENENTUKAN LOKASI FILE
folder_script = Path(__file__).resolve().parent
folder_root = folder_script.parents[1]
input_dir = folder_root / "stage-1" / "3-outlier-and-split-data" / "output"
output_dir = folder_script / "output"
output_dir.mkdir(exist_ok=True)

# 2. MENGAMBIL DATA
df_train = pd.read_csv(input_dir / "train-diabetes.csv")
df_test = pd.read_csv(input_dir / "test-diabetes.csv")

print("=== TAHAP 4: TRANSFORMASI DATA (Z-SCORE) ===")
print("Data train :", df_train.shape)
print("Data test  :", df_test.shape)

# 3. MEMISAHKAN NAMA FITUR DAN TARGET
TARGET_COL = "Outcome"
FEATURE_COLS = [kolom for kolom in df_train.columns if kolom != TARGET_COL]

# 4. MENGHITUNG MEAN DAN STD DARI DATA TRAIN
# Dihitung dari data train saja agar nilai pada data test tidak ikut memengaruhi
mean = df_train[FEATURE_COLS].mean()
std = df_train[FEATURE_COLS].std()

print("\n=== Mean tiap fitur (data train) ===")
print(mean.round(4))
print("\n=== Std tiap fitur (data train) ===")
print(std.round(4))

# 5. MENTRANSFORMASI DATA TRAIN DAN TEST
df_train_scaled = (df_train[FEATURE_COLS] - mean) / std
df_test_scaled = (df_test[FEATURE_COLS] - mean) / std

# Kolom Outcome adalah label kelas, jadi tidak ikut ditransformasi
df_train_scaled[TARGET_COL] = df_train[TARGET_COL]
df_test_scaled[TARGET_COL] = df_test[TARGET_COL]

print("\n=== 5 data train sebelum transformasi ===")
print(df_train.head())
print("\n=== 5 data train sesudah transformasi ===")
print(df_train_scaled.head())

print("\n=== Mean tiap fitur sesudah transformasi ===")
print(df_train_scaled[FEATURE_COLS].mean().round(4))
print("\n=== Std tiap fitur sesudah transformasi ===")
print(df_train_scaled[FEATURE_COLS].std().round(4))

# 6. MENYIMPAN HASIL KE FILE CSV
df_train_scaled.to_csv(output_dir / "train-transformed.csv", index=False)
df_test_scaled.to_csv(output_dir / "test-transformed.csv", index=False)

# 7. MEMBUAT GRAFIK PERBANDINGAN SKALA
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Perbandingan Skala Fitur Sebelum dan Sesudah Z-Score")

df_train[FEATURE_COLS].boxplot(ax=axes[0], rot=45)
axes[0].set_title("Sebelum Transformasi")
axes[0].set_ylabel("Nilai Asli")

df_train_scaled[FEATURE_COLS].boxplot(ax=axes[1], rot=45)
axes[1].set_title("Sesudah Transformasi")
axes[1].set_ylabel("Nilai Z-Score")

# Label dirapatkan ke kanan agar tetap lurus di bawah kotaknya
for ax in axes:
    for label in ax.get_xticklabels():
        label.set_ha("right")

plt.tight_layout()
plt.savefig(output_dir / "scaling-comparison.png", dpi=150)
plt.close()

print("\n=== SELESAI ===")
print("Hasil disimpan di folder output:")
print("- train-transformed.csv")
print("- test-transformed.csv")
print("- scaling-comparison.png")
