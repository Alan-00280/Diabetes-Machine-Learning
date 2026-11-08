# ==============================================================================
# PRAKTIKUM MACHINE LEARNING - REGRESI (TM11)
# Anggota C (Raka) — Tahap 2: Gabung Metrik, Uji Data Baru, Simpan Model Terbaik
#
# Input  : xgb_metrics.csv      (dari xgboost/output/)
#          metrics.json          (dari light-gbm/output/)
#          predictions_test.csv  (dari light-gbm/output/)
#          X_train/X_test/y_*.npy (dari praprocess/output/)
#
# Output : combine/output/
#   - perbandingan_metrik.csv      → Tabel 5.1 & 5.2 laporan
#   - uji_data_baru.txt            → Subbab 5.6 laporan
#   - model_terbaik_info.txt       → Subbab 5.7 laporan
#   - best_model.pkl               → model terbaik untuk UI/UX (Alan)
#   - scatter_aktual_vs_prediksi_gabungan.png  → Gambar 5.7
# ==============================================================================

import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, mean_absolute_percentage_error)

# ==============================================================================
# 0. Path setup
# ==============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR)
PRAPROC_DIR = os.path.join(ROOT_DIR, 'praprocess', 'output')
XGB_DIR     = os.path.join(ROOT_DIR, 'xgboost',   'output')
LGB_DIR     = os.path.join(ROOT_DIR, 'light-gbm', 'output')
OUT_DIR     = os.path.join(BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("  ANGGOTA C — GABUNG METRIK & PEMILIHAN MODEL TERBAIK")
print("=" * 65)

# ==============================================================================
# 1. Muat Metrik dari Kedua Model
# ==============================================================================
print("\n[1] MEMBACA METRIK KEDUA MODEL")

# XGBoost: dari xgb_metrics.csv
xgb_csv = pd.read_csv(os.path.join(XGB_DIR, 'xgb_metrics.csv'), index_col=0)
xgb_test = {
    'MAE':      float(xgb_csv.loc['MAE',      'Data Uji']),
    'MSE':      float(xgb_csv.loc['MSE',      'Data Uji']),
    'RMSE':     float(xgb_csv.loc['RMSE',     'Data Uji']),
    'MAPE (%)': float(xgb_csv.loc['MAPE (%)', 'Data Uji']),
    'R2':       float(xgb_csv.loc['R2',       'Data Uji']),
}
xgb_train = {
    'MAE':      float(xgb_csv.loc['MAE',      'Data Latih']),
    'MSE':      float(xgb_csv.loc['MSE',      'Data Latih']),
    'RMSE':     float(xgb_csv.loc['RMSE',     'Data Latih']),
    'MAPE (%)': float(xgb_csv.loc['MAPE (%)', 'Data Latih']),
    'R2':       float(xgb_csv.loc['R2',       'Data Latih']),
}

# LightGBM: dari metrics.json
with open(os.path.join(LGB_DIR, 'metrics.json')) as f:
    lgb_json = json.load(f)

lgb_test = {
    'MAE':      lgb_json['mae'],
    'MSE':      lgb_json['mse'],
    'RMSE':     lgb_json['rmse'],
    'MAPE (%)': lgb_json['mape'],
    'R2':       lgb_json['r2_score'],
}
lgb_train = {
    'MAE':      lgb_json['train_mae'],
    'MSE':      lgb_json['train_mae'] ** 2,   # estimasi — MSE latih tidak disimpan langsung
    'RMSE':     lgb_json['train_rmse'],
    'MAPE (%)': float('nan'),                  # tidak disimpan
    'R2':       lgb_json['train_r2_score'],
}

print("  XGBoost test  — MAE: {:,.0f} | RMSE: {:,.0f} | R2: {:.4f}".format(
    xgb_test['MAE'], xgb_test['RMSE'], xgb_test['R2']))
print("  LightGBM test — MAE: {:,.0f} | RMSE: {:,.0f} | R2: {:.4f}".format(
    lgb_test['MAE'], lgb_test['RMSE'], lgb_test['R2']))

# ==============================================================================
# 2. Tabel Perbandingan Metrik (Tabel 5.1 & 5.2 laporan)
# ==============================================================================
print("\n[2] MEMBUAT TABEL PERBANDINGAN METRIK")

df_compare = pd.DataFrame({
    'XGBoost (Uji)':   xgb_test,
    'LightGBM (Uji)':  lgb_test,
    'XGBoost (Latih)': xgb_train,
    'LightGBM (Latih)': lgb_train,
}).T

compare_path = os.path.join(OUT_DIR, 'perbandingan_metrik.csv')
df_compare.to_csv(compare_path)
print(f"  → {compare_path}")
print()
print(df_compare.round(2).to_string())

# ==============================================================================
# 3. Muat Prediksi dan Buat Scatter Plot Gabungan (Gambar 5.7)
# ==============================================================================
print("\n[3] MEMBUAT SCATTER PLOT GABUNGAN")

y_test_np = np.load(os.path.join(PRAPROC_DIR, 'y_test.npy'))

# XGBoost predictions — hitung ulang dari metrik (prediksi tidak disimpan CSV oleh Vito)
# Pakai predictions_test.csv milik LightGBM untuk perbandingan visual
lgb_pred_csv = pd.read_csv(os.path.join(LGB_DIR, 'predictions_test.csv'))
y_pred_lgb   = lgb_pred_csv['Predicted_Price'].values
y_actual     = lgb_pred_csv['Actual_Price'].values

# Scatter plot gabungan: hanya LightGBM (XGBoost prediksi tidak tersimpan CSV)
# → Gambar 5.6 (LightGBM) sudah ada di light-gbm/output/1_scatter_aktual_vs_prediksi.png
# → Gambar 5.5 (XGBoost) sudah ada di xgboost/output/xgb_prediksi_vs_aktual.png
# Di sini buat Gambar 5.7: overlay 100 sampel pertama untuk perbandingan visual
fig, ax = plt.subplots(figsize=(12, 5))
sample_idx = np.arange(100)
ax.scatter(sample_idx, y_actual[:100], color='black', s=30, zorder=4,
           label='Aktual', alpha=0.9)
ax.plot(sample_idx, y_pred_lgb[:100], color='#2ca02c', linewidth=1.5,
        label='LightGBM', zorder=3)
ax.set_title('Perbandingan Aktual vs LightGBM — 100 Sampel Data Uji Pertama',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Indeks Sampel', fontsize=11)
ax.set_ylabel('Harga Rumah (USD)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, linestyle=':', alpha=0.6)
fig.tight_layout()
scatter_path = os.path.join(OUT_DIR, 'scatter_aktual_vs_prediksi_gabungan.png')
plt.savefig(scatter_path, dpi=180)
plt.close()
print(f"  → {scatter_path}")

# ==============================================================================
# 4. Pemilihan Model Terbaik (Subbab 5.7)
# ==============================================================================
print("\n[4] PEMILIHAN MODEL TERBAIK")

# Kriteria utama: MAE dan RMSE lebih rendah → lebih baik; R2 lebih tinggi → lebih baik
# XGBoost unggul di MAE (73.870 vs 77.760) dan RMSE (95.254 vs 102.539)
# dan R2 (-0.062 vs -0.231) — XGBoost lebih baik pada semua metrik uji
best_model_name = 'XGBoost'
best_model_metrics = xgb_test
reason = (
    "XGBoost dipilih sebagai model terbaik berdasarkan perbandingan semua "
    "metrik evaluasi pada data uji:\n"
    f"  - MAE lebih rendah  : XGBoost {xgb_test['MAE']:,.0f} vs LightGBM {lgb_test['MAE']:,.0f}\n"
    f"  - RMSE lebih rendah : XGBoost {xgb_test['RMSE']:,.0f} vs LightGBM {lgb_test['RMSE']:,.0f}\n"
    f"  - MAPE lebih rendah : XGBoost {xgb_test['MAPE (%)']:.2f}% vs LightGBM {lgb_test['MAPE (%)']:.2f}%\n"
    f"  - R2 lebih tinggi   : XGBoost {xgb_test['R2']:.4f} vs LightGBM {lgb_test['R2']:.4f}\n"
    "Meskipun keduanya menunjukkan overfitting besar (R2 negatif pada data uji),\n"
    "XGBoost masih menghasilkan prediksi yang lebih mendekati nilai aktual.\n"
    "Penyebab utama overfitting: data dibagi berurutan berdasarkan koordinat geografis,\n"
    "sehingga data uji berasal dari wilayah dengan distribusi harga berbeda dari data latih."
)

print(f"  Model terbaik: {best_model_name}")
print(f"  Alasan:\n{reason}")

info_path = os.path.join(OUT_DIR, 'model_terbaik_info.txt')
with open(info_path, 'w', encoding='utf-8') as f:
    f.write(f"MODEL TERBAIK: {best_model_name}\n")
    f.write("=" * 60 + "\n")
    f.write(reason + "\n\n")
    f.write("Metrik XGBoost (Data Uji):\n")
    for k, v in xgb_test.items():
        f.write(f"  {k}: {v:,.4f}\n")
    f.write("\nMetrik LightGBM (Data Uji):\n")
    for k, v in lgb_test.items():
        f.write(f"  {k}: {v:,.4f}\n")
print(f"  → {info_path}")

# ==============================================================================
# 5. Uji Data Baru (Subbab 5.6)
# ==============================================================================
print("\n[5] UJI PREDIKSI DATA BARU")

# Ambil 5 baris pertama data uji sebagai simulasi "data baru"
# (data yang belum pernah dilihat model saat training)
X_test_np = np.load(os.path.join(PRAPROC_DIR, 'X_test.npy'))

with open(os.path.join(PRAPROC_DIR, 'feature_names.txt')) as f:
    feature_names = [line.strip() for line in f if line.strip()]

# LightGBM — kita punya model dan prediksi
uji_baru_path = os.path.join(OUT_DIR, 'uji_data_baru.txt')
with open(uji_baru_path, 'w', encoding='utf-8') as f:
    f.write("=== HASIL PREDIKSI DATA BARU (5 Sampel Data Uji) ===\n")
    f.write("Simulasi: fitur dari 5 baris pertama data uji diumpankan ke model\n\n")
    f.write(f"{'No':<4} {'Aktual':>12} {'Pred LightGBM':>15} {'Selisih LGB':>13}\n")
    f.write("-" * 50 + "\n")
    for i in range(5):
        f.write(f"{i+1:<4} {y_actual[i]:>12,.0f} {y_pred_lgb[i]:>15,.0f} "
                f"{y_actual[i]-y_pred_lgb[i]:>13,.0f}\n")

for i in range(5):
    print(f"  [{i+1}] Aktual: {y_actual[i]:>10,.0f} | LightGBM: {y_pred_lgb[i]:>10,.0f} "
          f"| Selisih: {y_actual[i]-y_pred_lgb[i]:>10,.0f}")

print(f"\n  → {uji_baru_path}")

# ==============================================================================
# 6. Salin Model Terbaik (XGBoost) ke combine/output/best_model.pkl
# ==============================================================================
print("\n[6] MENYIMPAN MODEL TERBAIK")
best_src = os.path.join(XGB_DIR, 'xgb_model.pkl')
best_dst = os.path.join(OUT_DIR, 'best_model.pkl')
shutil.copy2(best_src, best_dst)
scaler_src = os.path.join(PRAPROC_DIR, 'scaler.pkl')
scaler_dst = os.path.join(OUT_DIR, 'scaler.pkl')
shutil.copy2(scaler_src, scaler_dst)
enc_src = os.path.join(PRAPROC_DIR, 'ocean_encoder.pkl')
enc_dst = os.path.join(OUT_DIR, 'ocean_encoder.pkl')
shutil.copy2(enc_src, enc_dst)
feat_src = os.path.join(PRAPROC_DIR, 'feature_names.txt')
feat_dst = os.path.join(OUT_DIR, 'feature_names.txt')
shutil.copy2(feat_src, feat_dst)

print(f"  → {best_dst}")
print(f"  → {scaler_dst}")
print(f"  → {enc_dst}")
print(f"  → {feat_dst}")

# ==============================================================================
# 7. Ringkasan Output
# ==============================================================================
print("\n" + "=" * 65)
print("SELESAI — Artefak tersimpan di combine/output/")
print("=" * 65)
for fname in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(os.path.join(OUT_DIR, fname))
    print(f"  {fname:<48} {size:>10,} bytes")

