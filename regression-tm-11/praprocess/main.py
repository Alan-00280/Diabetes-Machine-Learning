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
print(f"Shape awal : {df.shape}")
print(f"Kolom      : {df.columns.tolist()}")

# ==============================================================================
# 2. EDA — Eksplorasi Awal
# ==============================================================================
print("\n" + "=" * 60)
print("2. EDA — EKSPLORASI AWAL")
print("=" * 60)

print("\n[2.1] Tipe data:")
print(df.dtypes)

print("\n[2.2] Statistik deskriptif (numerik):")
print(df.describe())

print("\n[2.3] Missing values:")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n[2.4] Distribusi ocean_proximity:")
print(df['ocean_proximity'].value_counts())

print("\n[2.5] Target — median_house_value:")
print(f"  min  : {df['median_house_value'].min():,.0f}")
print(f"  max  : {df['median_house_value'].max():,.0f}")
print(f"  mean : {df['median_house_value'].mean():,.0f}")
# 965 baris ada di batas atas 500.001 → nilai ter-cap oleh sumber data
n_capped = (df['median_house_value'] == 500001).sum()
print(f"  Baris dengan nilai ter-cap (500.001): {n_capped}")

# Simpan visualisasi distribusi target sebelum praproses
plt.figure(figsize=(8, 4))
df['median_house_value'].hist(bins=50, color='steelblue', edgecolor='white')
plt.title('Distribusi median_house_value (sebelum praproses)')
plt.xlabel('median_house_value')
plt.ylabel('Frekuensi')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2.1_distribusi_target_awal.png'), dpi=150)
plt.close()

# Heatmap korelasi numerik awal
plt.figure(figsize=(10, 7))
num_cols_init = df.select_dtypes(include='number').columns
sns.heatmap(df[num_cols_init].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Korelasi Fitur Numerik (sebelum praproses)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2.2_heatmap_korelasi_awal.png'), dpi=150)
plt.close()

print("\n  → Visualisasi EDA disimpan ke folder output/")

# ==============================================================================
# 3. Tangani Missing Value
# ==============================================================================
print("\n" + "=" * 60)
print("3. MISSING VALUE")
print("=" * 60)

# total_bedrooms: 207 baris null → imputasi dengan median
# (lebih robust terhadap distribusi skewed dibanding mean)
median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'] = df['total_bedrooms'].fillna(median_bedrooms)
print(f"  total_bedrooms: 207 null → diisi median = {median_bedrooms:.1f}")
print(f"  Missing setelah imputasi: {df.isnull().sum().sum()} total")

# ==============================================================================
# 4. Sorting
# ==============================================================================
print("\n" + "=" * 60)
print("4. SORTING")
print("=" * 60)

# Urutkan berdasarkan latitude lalu longitude (selatan→utara, barat→timur)
# agar urutan data deterministik dan reproducible untuk semua anggota tim
df = df.sort_values(by=['latitude', 'longitude']).reset_index(drop=True)
print(f"  Data diurutkan berdasarkan latitude, longitude (ascending)")
print(f"  Shape setelah sorting: {df.shape}")

# ==============================================================================
# 5. Feature Engineering
# ==============================================================================
print("\n" + "=" * 60)
print("5. FEATURE ENGINEERING")
print("=" * 60)

# Fitur turunan yang terbukti meningkatkan performa model housing
df['rooms_per_household']       = df['total_rooms']    / df['households']
df['bedrooms_per_room']         = df['total_bedrooms'] / df['total_rooms']
df['population_per_household']  = df['population']     / df['households']

print("  Fitur baru ditambahkan:")
print("  - rooms_per_household       = total_rooms / households")
print("  - bedrooms_per_room         = total_bedrooms / total_rooms")
print("  - population_per_household  = population / households")
print(f"  Shape setelah FE: {df.shape}")

# ==============================================================================
# 6. Tangani Outlier (IQR Capping / Winsorization)
# ==============================================================================
print("\n" + "=" * 60)
print("6. OUTLIER — IQR CAPPING")
print("=" * 60)

# Kolom yang di-cap: fitur turunan dan beberapa fitur asli yang sangat skewed
# longitude & latitude TIDAK di-cap (koordinat geografis bermakna)
# median_house_value TIDAK di-cap (target, capping akan bias model)
# housing_median_age TIDAK di-cap (bounded 1–52, wajar)
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
    print(f"  {col:<30}: {n_before:>4} outlier → cap [{lower:.2f}, {upper:.2f}]")

# Boxplot setelah capping
plt.figure(figsize=(14, 5))
df[OUTLIER_COLS].boxplot(rot=30, patch_artist=True)
plt.title('Boxplot Fitur Setelah Outlier Capping (IQR)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '6.1_boxplot_setelah_capping.png'), dpi=150)
plt.close()

# ==============================================================================
# 7. Encode Fitur Kategorik
# ==============================================================================
print("\n" + "=" * 60)
print("7. ENCODE FITUR KATEGORIK")
print("=" * 60)

# OrdinalEncoder dengan urutan berdasarkan jarak dari laut (intuisi domain):
# ISLAND=4, NEAR BAY=3, NEAR OCEAN=2, <1H OCEAN=1, INLAND=0
# Urutan ini mencerminkan "kedekatan ke laut" yang berkorelasi dengan harga
ocean_order = [['INLAND', '<1H OCEAN', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']]
enc = OrdinalEncoder(categories=ocean_order)
df['ocean_proximity_enc'] = enc.fit_transform(df[['ocean_proximity']])
df = df.drop(columns=['ocean_proximity'])

print(f"  ocean_proximity → ocean_proximity_enc (ordinal, skala 0–4)")
print(f"  Urutan: INLAND=0, <1H OCEAN=1, NEAR OCEAN=2, NEAR BAY=3, ISLAND=4")
print(f"  Shape setelah encoding: {df.shape}")

# ==============================================================================
# 8. Train-Test Split (80/20, urutan deterministik — TIDAK diacak)
# ==============================================================================
print("\n" + "=" * 60)
print("8. TRAIN-TEST SPLIT")
print("=" * 60)

TARGET = 'median_house_value'
FEATURE_COLS = [c for c in df.columns if c != TARGET]

X = df[FEATURE_COLS].values
y = df[TARGET].values

# Karena data sudah disortir (deterministik), split berdasarkan posisi
# 80% pertama = train, 20% terakhir = test
# Ini konsisten dengan TimeSeriesSplit yang dipakai Vito (A) dan Alan (B)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=False          # ← deterministik, tidak diacak
)

print(f"  Fitur         : {len(FEATURE_COLS)} kolom")
print(f"  X_train shape : {X_train.shape}")
print(f"  X_test  shape : {X_test.shape}")
print(f"  y_train range : [{y_train.min():,.0f}, {y_train.max():,.0f}]")
print(f"  y_test  range : [{y_test.min():,.0f}, {y_test.max():,.0f}]")
print(f"  Fitur: {FEATURE_COLS}")

# ==============================================================================
# 9. Scaling (StandardScaler — fit HANYA pada train)
# ==============================================================================
print("\n" + "=" * 60)
print("9. SCALING (StandardScaler)")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform pada train
X_test_scaled  = scaler.transform(X_test)        # transform saja pada test

print(f"  X_train_scaled — mean sample: {X_train_scaled.mean(axis=0)[:3].round(4)}")
print(f"  X_train_scaled — std  sample: {X_train_scaled.std(axis=0)[:3].round(4)}")
print("  Scaler di-fit HANYA pada data train (mencegah data leakage)")

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
with open(feature_names_path, 'w') as f:
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
with open(summary_path, 'w') as f:
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
    f.write(f"  total_bedrooms: 207 baris → imputasi median ({median_bedrooms:.1f})\n")
    f.write(f"\nOutlier capping (IQR):\n")
    for col, info in clip_log.items():
        f.write(f"  {col:<30}: {info['n_capped']:>4} baris → [{info['lower']:.2f}, {info['upper']:.2f}]\n")
    f.write(f"\nEncoding:\n")
    f.write(f"  ocean_proximity → ordinal (INLAND=0, <1H OCEAN=1, NEAR OCEAN=2, NEAR BAY=3, ISLAND=4)\n")
    f.write(f"\nSplit   : 80% train / 20% test, shuffle=False, random_state=42\n")
    f.write(f"Scaling : StandardScaler (fit pada train saja)\n")

print("  Artefak tersimpan:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size  = os.path.getsize(fpath)
    print(f"    {fname:<40} {size:>10,} bytes")

print("\n" + "=" * 60)
print("PRAPROSES SELESAI ✓")
print("=" * 60)
print("Langkah selanjutnya:")
print("  → Vito (XGBoost) & Alan (LightGBM) load X_train.npy, X_test.npy,")
print("    y_train.npy, y_test.npy dari praprocess/output/")
print("  → Feature names tersedia di praprocess/output/feature_names.txt")

