"""
PRAKTIKUM MACHINE LEARNING - REGRESI
Model XGBoost: California Housing Dataset
Anggota A (Vito) - TM11

Tugas:
  1. Inisialisasi model XGBoost
  2. Tuning hyperparameter dengan TimeSeriesSplit
  3. Training model terbaik
  4. Learning curve
  5. Feature importance
  6. Prediksi dan evaluasi pada data uji

Input  : regression-tm-11/praprocess/output/ (X_train, X_test, y_train, y_test)
Output : regression-tm-11/xgboost/output/
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# ==============================================================================
# 0. PATH SETUP
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'praprocess', 'output')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = 'median_house_value'
RANDOM_STATE = 42

# ==============================================================================
# 1. MEMBACA DATA HASIL PRAPROSES
# ==============================================================================
X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))[TARGET_COL]
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))[TARGET_COL]

print("=" * 70)
print("  XGBOOST - CALIFORNIA HOUSING (REGRESI)")
print("=" * 70)
print(f"\nData latih : {X_train.shape[0]} baris x {X_train.shape[1]} fitur")
print(f"Data uji   : {X_test.shape[0]} baris x {X_test.shape[1]} fitur")
print(f"Fitur      : {', '.join(X_train.columns)}")

# ==============================================================================
# 2. INISIALISASI DAN TUNING HYPERPARAMETER (TimeSeriesSplit)
# Data telah diurutkan berdasarkan indeks geografis dan dibagi berurutan 80/20,
# sehingga validasi silang memakai TimeSeriesSplit agar tidak memakai data
# masa depan untuk melatih data masa lalu.
# ==============================================================================
print("\n" + "=" * 70)
print("1. TUNING HYPERPARAMETER (GridSearchCV + TimeSeriesSplit)")
print("=" * 70)

model_awal = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

param_grid = {
    'n_estimators': [400, 800],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

tscv = TimeSeriesSplit(n_splits=5)
jumlah_kombinasi = 1
for nilai in param_grid.values():
    jumlah_kombinasi *= len(nilai)

print(f"\nJumlah kombinasi parameter : {jumlah_kombinasi}")
print(f"Skema validasi silang      : TimeSeriesSplit(n_splits={tscv.n_splits})")

pencarian = GridSearchCV(
    estimator=model_awal,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=0,
)
pencarian.fit(X_train, y_train)

print("\nParameter terbaik:")
for nama, nilai in pencarian.best_params_.items():
    print(f"  {nama:<20} : {nilai}")
print(f"RMSE validasi silang terbaik: {-pencarian.best_score_:.2f}")

# ==============================================================================
# 3. TRAINING MODEL TERBAIK
# ==============================================================================
print("\n" + "=" * 70)
print("2. TRAINING MODEL TERBAIK")
print("=" * 70)

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    **pencarian.best_params_,
)
model.fit(X_train, y_train)
print("Model selesai dilatih dengan parameter terbaik.")

# ==============================================================================
# 4. LEARNING CURVE
# RMSE data latih dan data uji pada tiap putaran pohon, untuk melihat
# apakah model mulai menghafal (overfitting) seiring bertambahnya pohon.
# ==============================================================================
print("\n" + "=" * 70)
print("3. LEARNING CURVE")
print("=" * 70)

model_riwayat = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='rmse',
    **pencarian.best_params_,
)
model_riwayat.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False,
)
riwayat = model_riwayat.evals_result()
rmse_latih = riwayat['validation_0']['rmse']
rmse_uji = riwayat['validation_1']['rmse']

print(f"RMSE akhir data latih : {rmse_latih[-1]:.2f}")
print(f"RMSE akhir data uji   : {rmse_uji[-1]:.2f}")
print(f"Jumlah pohon          : {len(rmse_latih)}")

plt.figure(figsize=(10, 6))
plt.plot(rmse_latih, label='Data Latih', linewidth=2)
plt.plot(rmse_uji, label='Data Uji', linewidth=2)
plt.xlabel('Jumlah Pohon (putaran boosting)')
plt.ylabel('RMSE')
plt.title('Learning Curve XGBoost')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_learning_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OUTPUT] xgb_learning_curve.png tersimpan")

# ==============================================================================
# 5. FEATURE IMPORTANCE
# ==============================================================================
print("\n" + "=" * 70)
print("4. FEATURE IMPORTANCE")
print("=" * 70)

penting = pd.Series(model.feature_importances_, index=X_train.columns)
penting_urut = penting.sort_values(ascending=True)

plt.figure(figsize=(10, 7))
penting_urut.plot(kind='barh', color='#4C72B0')
plt.xlabel('Skor Kepentingan Fitur')
plt.title('Feature Importance XGBoost')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OUTPUT] xgb_feature_importance.png tersimpan")

print("\nLima fitur paling penting:")
for nama, nilai in penting.sort_values(ascending=False).head(5).items():
    print(f"  {nama:<28} {nilai:.4f}")

# ==============================================================================
# 6. PREDIKSI DAN EVALUASI PADA DATA UJI
# ==============================================================================
print("\n" + "=" * 70)
print("5. PREDIKSI DAN EVALUASI")
print("=" * 70)

y_pred_latih = model.predict(X_train)
y_pred_uji = model.predict(X_test)


def hitung_metrik(aktual, prediksi):
    mse = mean_squared_error(aktual, prediksi)
    return {
        'MAE': mean_absolute_error(aktual, prediksi),
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2': r2_score(aktual, prediksi),
    }


metrik_latih = hitung_metrik(y_train, y_pred_latih)
metrik_uji = hitung_metrik(y_test, y_pred_uji)

tabel_metrik = pd.DataFrame({'Data Latih': metrik_latih, 'Data Uji': metrik_uji})
print()
print(tabel_metrik.round(2).to_string())

selisih_r2 = metrik_latih['R2'] - metrik_uji['R2']
print(f"\nSelisih R2 latih-uji : {selisih_r2:.4f}")
if selisih_r2 < 0.1:
    print("  -> gap kecil, model tidak menunjukkan overfitting yang berarti")
else:
    print("  -> gap cukup besar, perlu diwaspadai overfitting")

tabel_metrik.round(4).to_csv(os.path.join(OUTPUT_DIR, 'xgb_metrics.csv'))

# ==============================================================================
# 7. VISUALISASI PREDIKSI DAN RESIDUAL
# ==============================================================================
print("\n" + "=" * 70)
print("6. VISUALISASI PREDIKSI DAN RESIDUAL")
print("=" * 70)

# Grafik prediksi terhadap nilai aktual
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_uji, s=12, alpha=0.5, color='#4C72B0')
batas_min = min(y_test.min(), y_pred_uji.min())
batas_max = max(y_test.max(), y_pred_uji.max())
plt.plot([batas_min, batas_max], [batas_min, batas_max], 'r--', linewidth=2,
         label='Garis Prediksi Sempurna')
plt.xlabel('Nilai Aktual (median_house_value)')
plt.ylabel('Nilai Prediksi')
plt.title('Prediksi vs Aktual XGBoost pada Data Uji')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_prediksi_vs_aktual.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OUTPUT] xgb_prediksi_vs_aktual.png tersimpan")

# Grafik residual
residual = y_test.to_numpy() - y_pred_uji
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_uji, residual, s=12, alpha=0.5, color='#C44E52')
plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
plt.xlabel('Nilai Prediksi')
plt.ylabel('Residual (Aktual - Prediksi)')
plt.title('Plot Residual XGBoost pada Data Uji')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_residual.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OUTPUT] xgb_residual.png tersimpan")

print(f"\nResidual - rata-rata        : {residual.mean():.2f}")
print(f"Residual - standar deviasi  : {residual.std():.2f}")

# ==============================================================================
# 8. MENYIMPAN MODEL DAN RINGKASAN
# ==============================================================================
joblib.dump(model, os.path.join(OUTPUT_DIR, 'xgb_model.pkl'))
print("\n[OUTPUT] xgb_model.pkl tersimpan")

with open(os.path.join(OUTPUT_DIR, 'evaluasi_model_xgboost.txt'), 'w') as f:
    f.write("=== RINGKASAN EVALUASI XGBOOST (TM11) ===\n")
    f.write("Dataset: housing.csv (California Housing)\n")
    f.write(f"Jumlah Data Latih: {len(X_train)}\n")
    f.write(f"Jumlah Data Uji  : {len(X_test)}\n")
    f.write(f"Jumlah Fitur     : {X_train.shape[1]}\n\n")
    f.write("Parameter Terbaik:\n")
    for nama, nilai in pencarian.best_params_.items():
        f.write(f"  {nama}: {nilai}\n")
    f.write("  skema tuning: GridSearchCV + TimeSeriesSplit(n_splits=5)\n")
    f.write(f"  RMSE validasi silang terbaik: {-pencarian.best_score_:.2f}\n\n")
    f.write("Metrik Evaluasi:\n")
    f.write(tabel_metrik.round(4).to_string() + "\n\n")
    f.write("Feature Importance (tertinggi):\n")
    for nama, nilai in penting.sort_values(ascending=False).items():
        f.write(f"  {nama}: {nilai:.4f}\n")
    f.write(f"\nResidual - rata-rata: {residual.mean():.2f}\n")
    f.write(f"Residual - standar deviasi: {residual.std():.2f}\n")
print("[OUTPUT] evaluasi_model_xgboost.txt tersimpan")

print("\n=== SELESAI ===")
print(f"Semua output tersimpan di: {OUTPUT_DIR}")
