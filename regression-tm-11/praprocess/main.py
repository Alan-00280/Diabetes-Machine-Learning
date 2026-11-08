# ==============================================================================
# PRAKTIKUM MACHINE LEARNING - REGRESI
# Praproses Data: California Housing Dataset
# Anggota C (Raka) — TM11
#
# Tugas:
#   1. Load data
#   2. EDA (eksplorasi awal)
#   3. Tangani missing value
#   4. Sorting (urut berdasarkan indeks geografis untuk reproduktifitas)
#   5. Deteksi dan tangani outlier (IQR capping)
#   6. Feature engineering (rooms_per_household, bedrooms_per_room, population_per_household)
#   7. Encode fitur kategorik (OrdinalEncoder ocean_proximity)
#   8. Train-test split (TimeSeriesSplit-style: 80/20 urut, bukan acak) ← konsisten dgn A & B
#   9. Scaling (StandardScaler fit pada train, transform keduanya)
#  10. Simpan artefak: X_train, X_test, y_train, y_test, scaler, feature_names
#      → semua ke folder output/ agar A & B bisa langsung load
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import joblib

# ==============================================================================
# 0. Path setup
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)          # regression-tm-11/
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_PATH = os.path.join(ROOT_DIR, 'housing.csv')

# ==============================================================================
# 1. Load data
# ==============================================================================
print("=" * 60)
print("1. LOAD DATA")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n--- Output df.head() (5 Baris Pertama) ---")
print(df.head())

print(f"\n--- Output df.shape (Ukuran Dataset) ---")
print(f"Dimensi data (baris, kolom): {df.shape}")
print(f"Jumlah baris: {df.shape[0]:,} | Jumlah kolom: {df.shape[1]}")
print(f"Daftar kolom: {df.columns.tolist()}")

# ==============================================================================
# 2. EDA — Eksplorasi Awal
# ==============================================================================
print("\n" + "=" * 65)
print("4.1.2 EKSPLORASI DATA (EDA) — [Gambar 4.3]")
print("=" * 65)

print("\n--- Kode: df.info() ---")
df.info()

print("\n--- Kode: df.describe() (Statistik Deskriptif) ---")
print(df.describe().round(2))

# ==============================================================================
# 3. Tangani Missing Value dan Duplikasi
# ==============================================================================
print("\n" + "=" * 65)
print("4.1.3 PENGECEKAN MISSING VALUE & DUPLIKASI — [Gambar 4.5]")
print("=" * 65)

print("\n--- Kode: df.isnull().sum() (Pengecekan Nilai Hilang) ---")
print(df.isnull().sum())

print("\n--- Kode: df.duplicated().sum() (Pengecekan Duplikasi) ---")
n_duplikat = df.duplicated().sum()
print(f"Jumlah baris duplikat: {n_duplikat}")

print("\n--- Penanganan Missing Value (Imputasi Median) ---")
median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'] = df['total_bedrooms'].fillna(median_bedrooms)
print(f"Kode: df['total_bedrooms'].fillna({median_bedrooms:.1f})")
print(f"Nilai hilang total_bedrooms setelah imputasi: {df['total_bedrooms'].isnull().sum()}")
print(f"Total nilai hilang seluruh dataset: {df.isnull().sum().sum()}")

# Visualisasi EDA awal
print("\n[Simpan Visualisasi Target & Korelasi...]")
plt.figure(figsize=(8, 4))
df['median_house_value'].hist(bins=50, color='steelblue', edgecolor='white')
plt.title('Distribusi median_house_value (sebelum praproses)')
plt.xlabel('median_house_value')
plt.ylabel('Frekuensi')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2.1_distribusi_target_awal.png'), dpi=150)
plt.close()

plt.figure(figsize=(10, 7))
num_cols_init = df.select_dtypes(include='number').columns
sns.heatmap(df[num_cols_init].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Korelasi Fitur Numerik (sebelum praproses)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2.2_heatmap_korelasi_awal.png'), dpi=150)
plt.close()

# ==============================================================================
# 4. Sorting & Feature Engineering
# ==============================================================================
print("\n" + "=" * 65)
print("4.1.4 PENGURUTAN DATA & FEATURE ENGINEERING — [Gambar 4.6]")
print("=" * 65)

# Pengurutan geografis
df = df.sort_values(by=['latitude', 'longitude']).reset_index(drop=True)
print("\n--- Kode: Pengurutan Kronologis/Geografis ---")
print("df = df.sort_values(by=['latitude', 'longitude']).reset_index(drop=True)")
print(f"Shape data setelah sorting: {df.shape}")
print("Cuplikan koordinat setelah diurutkan (arah selatan ke utara):")
print(df[['latitude', 'longitude', 'housing_median_age', 'median_house_value']].head(3))

# Feature engineering
df['rooms_per_household']       = df['total_rooms']    / df['households']
df['bedrooms_per_room']         = df['total_bedrooms'] / df['total_rooms']
df['population_per_household']  = df['population']     / df['households']

print("\n--- Kode: Rekayasa Fitur Rasio ---")
print("df['rooms_per_household']      = df['total_rooms'] / df['households']")
print("df['bedrooms_per_room']        = df['total_bedrooms'] / df['total_rooms']")
print("df['population_per_household'] = df['population'] / df['households']")
print(f"Shape dataset setelah penambahan fitur: {df.shape}")
print("Cuplikan 3 baris fitur baru:")
print(df[['rooms_per_household', 'bedrooms_per_room', 'population_per_household']].head(3).round(3))

# ==============================================================================
# 5. Outlier Capping (IQR)
# ==============================================================================
print("\n" + "=" * 65)
print("4.1.5 PENANGANAN OUTLIER (IQR CAPPING)")
print("=" * 65)

OUTLIER_COLS = [
    'total_rooms', 'total_bedrooms', 'population', 'households',
    'median_income',
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
]

clip_log = {}
for col in OUTLIER_COLS:
    q1  = df[col].quantile(0.25)
    q3  = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_before = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    clip_log[col] = {'lower': lower, 'upper': upper, 'n_capped': int(n_before)}
    print(f"  {col:<28}: {n_before:>4} outlier -> cap [{lower:.2f}, {upper:.2f}]")

# Boxplot setelah capping
plt.figure(figsize=(14, 5))
df[OUTLIER_COLS].boxplot(rot=30, patch_artist=True)
plt.title('Boxplot Fitur Setelah Outlier Capping (IQR)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '6.1_boxplot_setelah_capping.png'), dpi=150)
plt.close()

# ==============================================================================
# 6. Encode Kategorik & Train-Test Split
# ==============================================================================
print("\n" + "=" * 65)
print("4.1.6 PEMBAGIAN DATA (TRAIN-TEST SPLIT) — [Gambar 4.8]")
print("=" * 65)

# Encode ordinal
ocean_order = [['INLAND', '<1H OCEAN', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']]
enc = OrdinalEncoder(categories=ocean_order)
df['ocean_proximity_enc'] = enc.fit_transform(df[['ocean_proximity']])
df = df.drop(columns=['ocean_proximity'])

print("--- Encoding ocean_proximity (Ordinal 0-4) ---")
print("Urutan: INLAND=0, <1H OCEAN=1, NEAR OCEAN=2, NEAR BAY=3, ISLAND=4")

TARGET = 'median_house_value'
FEATURE_COLS = [c for c in df.columns if c != TARGET]

X = df[FEATURE_COLS].values
y = df[TARGET].values

# Split sekuensial (shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=False
)

print("\n--- Kode: train_test_split(test_size=0.2, shuffle=False) ---")
print(f"Ukuran X_train : {X_train.shape} (80% data latih)")
print(f"Ukuran X_test  : {X_test.shape}  (20% data uji)")
print(f"Ukuran y_train : {y_train.shape}")
print(f"Ukuran y_test  : {y_test.shape}")
print(f"Jumlah Fitur   : {len(FEATURE_COLS)} kolom")
print(f"Daftar Fitur   : {FEATURE_COLS}")

# ==============================================================================
# 7. Normalisasi / Scaling
# ==============================================================================
print("\n" + "=" * 65)
print("4.1.7 NORMALISASI / SCALING DATA — [Gambar 4.10]")
print("=" * 65)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit hanya pada train
X_test_scaled  = scaler.transform(X_test)        # transform pada test

print("--- Kode: StandardScaler (Fit hanya pada Data Latih) ---")
print("scaler = StandardScaler()")
print("X_train_scaled = scaler.fit_transform(X_train)")
print("X_test_scaled  = scaler.transform(X_test)")
print(f"\nRata-rata (mean) data latih setelah scaling: {X_train_scaled.mean(axis=0).round(4)[:4]} ...")
print(f"Standar deviasi (std) data latih setelah scaling: {X_train_scaled.std(axis=0).round(4)[:4]} ...")
print("Status: Scaler berhasil di-fit HANYA pada data latih (mencegah data leakage)")

# ==============================================================================
# 10. Simpan Artefak
# ==============================================================================
print("\n" + "=" * 60)
print("10. SIMPAN ARTEFAK")
print("=" * 60)

# Data split (numpy array)
np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train_scaled)
np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'),  X_test_scaled)
np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'),  y_test)

# Scaler dan encoder (untuk inverse transform / deploy)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
joblib.dump(enc,    os.path.join(OUTPUT_DIR, 'ocean_encoder.pkl'))

# Nama fitur (penting untuk feature importance A & B)
feature_names_path = os.path.join(OUTPUT_DIR, 'feature_names.txt')
with open(feature_names_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(FEATURE_COLS))

# Simpan juga versi CSV untuk inspeksi manual
pd.DataFrame(X_train_scaled, columns=FEATURE_COLS).to_csv(
    os.path.join(OUTPUT_DIR, 'X_train.csv'), index=False
)
pd.DataFrame(X_test_scaled, columns=FEATURE_COLS).to_csv(
    os.path.join(OUTPUT_DIR, 'X_test.csv'), index=False
)
pd.Series(y_train, name=TARGET).to_csv(
    os.path.join(OUTPUT_DIR, 'y_train.csv'), index=False
)
pd.Series(y_test, name=TARGET).to_csv(
    os.path.join(OUTPUT_DIR, 'y_test.csv'), index=False
)

# Ringkasan praproses
summary_path = os.path.join(OUTPUT_DIR, 'praprocess_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=== RINGKASAN PRAPROSES — California Housing (TM11) ===\n")
    f.write(f"Dataset         : housing.csv\n")
    f.write(f"Shape awal      : (20640, 10)\n")
    f.write(f"Shape akhir X   : {X_train_scaled.shape[1]} fitur\n")
    f.write(f"Jumlah train    : {len(X_train)} baris\n")
    f.write(f"Jumlah test     : {len(X_test)} baris\n")
    f.write(f"\nFitur akhir:\n")
    for i, col in enumerate(FEATURE_COLS):
        f.write(f"  [{i:2d}] {col}\n")
    f.write(f"\nMissing value:\n")
    f.write(f"  total_bedrooms: 207 baris -> imputasi median ({median_bedrooms:.1f})\n")
    f.write(f"\nOutlier capping (IQR):\n")
    for col, info in clip_log.items():
        f.write(f"  {col:<30}: {info['n_capped']:>4} baris -> [{info['lower']:.2f}, {info['upper']:.2f}]\n")
    f.write(f"\nEncoding:\n")
    f.write(f"  ocean_proximity -> ordinal (INLAND=0, <1H OCEAN=1, NEAR OCEAN=2, NEAR BAY=3, ISLAND=4)\n")
    f.write(f"\nSplit   : 80% train / 20% test, shuffle=False, random_state=42\n")
    f.write(f"Scaling : StandardScaler (fit pada train saja)\n")

print("  Artefak tersimpan:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size  = os.path.getsize(fpath)
    print(f"    {fname:<40} {size:>10,} bytes")

print("\n" + "=" * 60)
print("PRAPROSES SELESAI [OK]")
print("=" * 60)
print("Langkah selanjutnya:")
print("  -> Vito (XGBoost) & Alan (LightGBM) load X_train.npy, X_test.npy,")
print("    y_train.npy, y_test.npy dari praprocess/output/")
print("  -> Feature names tersedia di praprocess/output/feature_names.txt")

