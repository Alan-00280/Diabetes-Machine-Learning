# ==============================================================================
# PRAKTIKUM MACHINE LEARNING - REGRESI
# Algoritma: LightGBM (Light Gradient Boosting Machine)
# Anggota: Alan — TM11 (Kelompok Ganjil)
# Dataset: California Housing (data sekunder dari praprocess)
#
# Tugas Sesuai Modul TM11:
#   1. Prediksi harga menggunakan LightGBM
#   2. Load data hasil praproses (X_train, X_test, y_train, y_test)
#   3. Evaluasi wajib 5 metrik: MAE, MAPE, RMSE, MSE, dan R2
#   4. Visualisasi hasil prediksi dan analisis performa
#   5. Simpan model dan artefak untuk laporan dan integrasi UI/UX
# ==============================================================================

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMRegressor, plot_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# ==============================================================================
# 0. Path Setup
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PRAPROCESS_OUT = os.path.join(ROOT_DIR, 'praprocess', 'output')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("     PELATIHAN & EVALUASI MODEL REGRESI: LIGHTGBM (KELOMPOK GANJIL)     ")
print("=" * 70)

# ==============================================================================
# 1. Load Data Hasil Praproses
# ==============================================================================
print("\n[1] MEMUAT DATASET DARI PRAPROSES...")

X_train = np.load(os.path.join(PRAPROCESS_OUT, 'X_train.npy'))
X_test  = np.load(os.path.join(PRAPROCESS_OUT, 'X_test.npy'))
y_train = np.load(os.path.join(PRAPROCESS_OUT, 'y_train.npy'))
y_test  = np.load(os.path.join(PRAPROCESS_OUT, 'y_test.npy'))

feature_names_file = os.path.join(PRAPROCESS_OUT, 'feature_names.txt')
with open(feature_names_file, 'r') as f:
    feature_names = [line.strip() for line in f if line.strip()]

print(f"  * X_train shape : {X_train.shape}")
print(f"  * X_test shape  : {X_test.shape}")
print(f"  * y_train shape : {y_train.shape} (Rentang: ${y_train.min():,.0f} - ${y_train.max():,.0f})")
print(f"  * y_test shape  : {y_test.shape}  (Rentang: ${y_test.min():,.0f} - ${y_test.max():,.0f})")
print(f"  * Fitur ({len(feature_names)}): {', '.join(feature_names)}")

# ==============================================================================
# 2. Inisialisasi & Konfigurasi Hyperparameter LightGBM
# ==============================================================================
print("\n[2] MEMBANGUN DAN MELATIH MODEL LIGHTGBM...")
# Karakteristik LightGBM:
# - Leaf-wise tree growth (menghasilkan reduksi loss optimal)
# - Histogram-based algorithm (komputasi sangat cepat dan hemat memori)
# - Regularisasi l1/l2 untuk kontrol overfitting

# Split sebagian data train untuk evaluasi kurva validasi per iterasi
val_size = int(len(X_train) * 0.15)
X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

lgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'importance_type': 'gain'
}

model = LGBMRegressor(**lgb_params)

start_time = time.time()
model.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val)],
    eval_names=['Train', 'Validation'],
    eval_metric=['rmse', 'mae']
)
training_time = time.time() - start_time
print(f"  * Pelatihan selesai dalam: {training_time:.2f} detik")

# Simpan model final yang dilatih pada seluruh data train
final_model = LGBMRegressor(**lgb_params)
final_model.fit(X_train, y_train)
model_path = os.path.join(OUTPUT_DIR, 'lightgbm_model.pkl')
joblib.dump(final_model, model_path)
print(f"  * Model tersimpan di: {model_path}")

# ==============================================================================
# 3. Prediksi dan Pengujian (Testing)
# ==============================================================================
print("\n[3] PENGUJIAN DAN PREDIKSI DATA UJI (TEST SET)...")
y_pred_train = final_model.predict(X_train)
y_pred_test  = final_model.predict(X_test)

# ==============================================================================
# 4. Evaluasi 5 Metrik Wajib Modul TM11
# ==============================================================================
# 1. MAE (Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred_test)

# 2. MSE (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred_test)

# 3. RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)

# 4. MAPE (Mean Absolute Percentage Error)
mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100

# 5. R2 Score (Koefisien Determinasi)
r2 = r2_score(y_test, y_pred_test)

# Metrik Train untuk analisa generalisasi model
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)

print("\n" + "=" * 50)
print("       HASIL EVALUASI MODEL LIGHTGBM (TEST SET)   ")
print("=" * 50)
print(f" 1. MAE  (Mean Absolute Error)            : ${mae:,.2f}")
print(f" 2. MSE  (Mean Squared Error)             : {mse:,.2f}")
print(f" 3. RMSE (Root Mean Squared Error)        : ${rmse:,.2f}")
print(f" 4. MAPE (Mean Absolute Percentage Error) : {mape:.2f}%")
print(f" 5. R² Score (Akurasi Varians Model)      : {r2:.4f} ({r2*100:.2f}%)")
print("=" * 50)
print(f" Perbandingan Train vs Test:")
print(f"  - Train R² : {train_r2:.4f} | Test R² : {r2:.4f}")
print(f"  - Train MAE: ${train_mae:,.2f} | Test MAE: ${mae:,.2f}")
print(f"  - Train RMSE: ${train_rmse:,.2f} | Test RMSE: ${rmse:,.2f}")
print("=" * 50)

# Simpan metrik ke JSON dan teks
metrics = {
    'model_name': 'LightGBM Regressor',
    'mae': float(mae),
    'mse': float(mse),
    'rmse': float(rmse),
    'mape': float(mape),
    'r2_score': float(r2),
    'train_mae': float(train_mae),
    'train_rmse': float(train_rmse),
    'train_r2_score': float(train_r2),
    'training_time_seconds': float(training_time)
}

with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

with open(os.path.join(OUTPUT_DIR, 'evaluasi_laporan.txt'), 'w') as f:
    f.write("=== LAPORAN EVALUASI MODEL REGRESI LIGHTGBM ===\n")
    f.write(f"Waktu Training : {training_time:.2f} detik\n")
    f.write(f"1. MAE         : ${mae:,.2f}\n")
    f.write(f"2. MSE         : {mse:,.2f}\n")
    f.write(f"3. RMSE        : ${rmse:,.2f}\n")
    f.write(f"4. MAPE        : {mape:.2f}%\n")
    f.write(f"5. R2 Score    : {r2:.4f} ({r2*100:.2f}%)\n\n")
    f.write("Performa Train vs Test:\n")
    f.write(f"  - Train R2   : {train_r2:.4f}\n")
    f.write(f"  - Test R2    : {r2:.4f}\n")
    f.write(f"  - Gap R2     : {abs(train_r2 - r2):.4f} (Model Seimbang / Tidak Overfit Ekstrem)\n")

# Simpan prediksi CSV untuk perbandingan
df_pred = pd.DataFrame({
    'Actual_Price': y_test,
    'Predicted_Price': y_pred_test,
    'Residual_Error': y_test - y_pred_test,
    'Absolute_Percentage_Error_%': np.abs((y_test - y_pred_test) / y_test) * 100
})
df_pred.to_csv(os.path.join(OUTPUT_DIR, 'predictions_test.csv'), index=False)

# ==============================================================================
# 5. Visualisasi Hasil & Evaluasi (Untuk Screenshot Laporan)
# ==============================================================================
print("\n[5] MEMBUAT DAN MENYIMPAN GRAFIK VISUALISASI...")

# Grafik 1: Scatter Plot Aktual vs Prediksi (Sesuai Modul Praktikum Bagian 5)
plt.figure(figsize=(9, 7))
plt.scatter(y_test, y_pred_test, alpha=0.3, color='#1f77b4', edgecolors='none', s=25, label='Data Uji')
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Prediksi Sempurna (y = x)')
plt.title(f'LightGBM: Nilai Aktual vs Nilai Prediksi\n(R² = {r2:.4f}, RMSE = ${rmse:,.0f})', fontsize=13, fontweight='bold')
plt.xlabel('Harga Rumah Aktual (USD)', fontsize=11)
plt.ylabel('Harga Rumah Prediksi (USD)', fontsize=11)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1_scatter_aktual_vs_prediksi.png'), dpi=200)
plt.close()
print("  * Tersimpan: 1_scatter_aktual_vs_prediksi.png")

# Grafik 2: Perbandingan Runut / Sampel Data Aktual vs Prediksi (100 Sampel Pertama)
plt.figure(figsize=(12, 5))
sample_range = slice(0, 100)
sample_idx = np.arange(100)
plt.scatter(sample_idx, y_test[sample_range], color='blue', label='Data Aktual', s=35, zorder=3)
plt.plot(sample_idx, y_pred_test[sample_range], color='crimson', linewidth=2, label='Prediksi LightGBM', zorder=2)
plt.title('Perbandingan Data Aktual vs Prediksi LightGBM (100 Sampel Data Uji Pertama)', fontsize=13, fontweight='bold')
plt.xlabel('Indeks Sampel', fontsize=11)
plt.ylabel('Harga Rumah (USD)', fontsize=11)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2_perbandingan_100_sampel.png'), dpi=200)
plt.close()
print("  * Tersimpan: 2_perbandingan_100_sampel.png")

# Grafik 3: Distribusi Residual / Error (y_test - y_pred)
residuals = y_test - y_pred_test
plt.figure(figsize=(9, 5))
sns.histplot(residuals, bins=50, kde=True, color='teal')
plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
plt.title('Distribusi Residual Error (Aktual - Prediksi)', fontsize=13, fontweight='bold')
plt.xlabel('Residual Error (USD)', fontsize=11)
plt.ylabel('Frekuensi', fontsize=11)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '3_distribusi_residual.png'), dpi=200)
plt.close()
print("  * Tersimpan: 3_distribusi_residual.png")

# Grafik 4: Feature Importance (Berdasarkan Gain / Kontribusi Prediksi)
importances = final_model.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='barh', color='#2ca02c')
plt.title('Feature Importance LightGBM (Berdasarkan Gain Reduksi Loss)', fontsize=13, fontweight='bold')
plt.xlabel('Total Gain', fontsize=11)
plt.ylabel('Fitur', fontsize=11)
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4_feature_importance.png'), dpi=200)
plt.close()
print("  * Tersimpan: 4_feature_importance.png")

# Grafik 5: Learning Curve Evaluasi Iterasi Boosting (Train vs Validation Loss)
eval_results = model.evals_result_
plt.figure(figsize=(10, 5))
plt.plot(eval_results['Train']['rmse'], label='Train RMSE', color='blue', alpha=0.8)
plt.plot(eval_results['Validation']['rmse'], label='Validation RMSE', color='orange', alpha=0.8)
plt.title('Learning Curve LightGBM: RMSE per Iterasi Boosting', fontsize=13, fontweight='bold')
plt.xlabel('Iterasi Boosting (Trees)', fontsize=11)
plt.ylabel('RMSE (USD)', fontsize=11)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5_learning_curve.png'), dpi=200)
plt.close()
print("  * Tersimpan: 5_learning_curve.png")

# ==============================================================================
# 6. Contoh Prediksi Data Baru (Simulasi Deployment)
# ==============================================================================
print("\n[6] SIMULASI PREDIKSI DATA BARU:")
sample_input = X_test[0:1]
sample_actual = y_test[0]
sample_pred = final_model.predict(sample_input)[0]

print(f"  * Nilai Harga Rumah Aktual  : ${sample_actual:,.2f}")
print(f"  * Hasil Prediksi LightGBM   : ${sample_pred:,.2f}")
print(f"  * Selisih (Error)           : ${abs(sample_actual - sample_pred):,.2f} ({abs(sample_actual - sample_pred)/sample_actual*100:.2f}%)")

print("\n" + "=" * 70)
print("EKSEKUSI PROGRAM LIGHTGBM SELESAI DENGAN SUKSES ✓")
print("Semua grafik visualisasi dan metrik telah tersimpan di folder 'output/'")
print("=" * 70)

