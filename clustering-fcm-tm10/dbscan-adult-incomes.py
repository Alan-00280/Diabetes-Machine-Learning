# ==============================================================================
# PRAKTIKUM MACHINE LEARNING - CLUSTERING MENGGUNAKAN DBSCAN
# Dataset: adult-incomes-dataset.csv
# Anggota C (Raka) — TM10
# ==============================================================================

# ==============================================================================
# 4.1 Persiapan Data (konsisten dengan FCM adult-incomes.py)
# ==============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output-visualisasi-dan-eksplorasi-adult-incomes')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
dataset_path = os.path.join(BASE_DIR, 'adult-incomes-dataset.csv')
df = pd.read_csv(dataset_path)
print("=== DBSCAN — Adult Incomes Dataset ===")
print("Shape:", df.shape)

# Handle missing values ('?' → NaN → imputasi modus)
df = df.replace('?', np.nan)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Penanganan outlier: capping IQR (konsisten dengan adult-incomes.py)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

# Fitur numerik yang sama dengan FCM
fitur = ['age', 'educational-num', 'hours-per-week']
X = df[fitur].values

# Standarisasi (wajib untuk DBSCAN berbasis jarak Euclidean)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Fitur DBSCAN  : {fitur}")
print(f"Jumlah data   : {len(df)}")
print(f"Shape X_scaled: {X_scaled.shape}")


# ==============================================================================
# 4.3 Penentuan Parameter DBSCAN — k-Distance Graph
# ==============================================================================
print("\n=== 4.3 Penentuan Parameter DBSCAN (k-Distance Graph) ===")

# MinPts standar = 2 * n_dimensi = 2 * 3 = 6
# Untuk dataset besar, lebih baik gunakan minPts sedikit lebih besar (misal 10)
k_neighbors = 6

# Karena dataset besar (48k baris), sampling 5000 titik untuk mempercepat
np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), size=5000, replace=False)
X_sample = X_scaled[sample_idx]

nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean').fit(X_sample)
distances, _ = nbrs.kneighbors(X_sample)

k_distances = distances[:, k_neighbors - 1]
k_distances_sorted = np.sort(k_distances)[::-1]

# Hitung persentil untuk panduan eps
p90 = np.percentile(k_distances, 90)
p95 = np.percentile(k_distances, 95)
p75 = np.percentile(k_distances, 75)
print(f"  Percentile 75%: {p75:.3f}")
print(f"  Percentile 90%: {p90:.3f}")
print(f"  Percentile 95%: {p95:.3f}")

plt.figure(figsize=(11, 5))
plt.plot(k_distances_sorted, color='steelblue', linewidth=1.5)
plt.axhline(y=p75, color='green',  linestyle='--', linewidth=1.5, label=f'P75 = {p75:.2f}')
plt.axhline(y=p90, color='orange', linestyle='--', linewidth=1.5, label=f'P90 = {p90:.2f}')
plt.axhline(y=p95, color='red',    linestyle='--', linewidth=1.5, label=f'P95 = {p95:.2f}')
plt.xlabel('Data Points Sampel (diurutkan)')
plt.ylabel(f'Jarak ke-{k_neighbors} Tetangga Terdekat')
plt.title(f'k-Distance Graph (k={k_neighbors}, sample=5000) — Adult Incomes')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dbscan_kdistance_graph.png'), dpi=150)
print("Simpan: dbscan_kdistance_graph.png")


# ==============================================================================
# Tuning eps dan minPts
# ==============================================================================
print("\n=== Tuning eps dan minPts (Adult Incomes) ===")
print("(menggunakan subset 10k data untuk efisiensi komputasi)")

# Gunakan subset untuk tuning
np.random.seed(42)
tune_idx = np.random.choice(len(X_scaled), size=10000, replace=False)
X_tune   = X_scaled[tune_idx]

eps_values    = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
minpts_values = [5, 6, 8, 10, 15]

results = []
for eps in eps_values:
    for minpts in minpts_values:
        db = DBSCAN(eps=eps, min_samples=minpts, metric='euclidean')
        labels = db.fit_predict(X_tune)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = np.sum(labels == -1)
        pct_noise  = n_noise / len(labels) * 100

        if n_clusters >= 2 and n_noise < len(labels):
            non_noise_mask = labels != -1
            sil = silhouette_score(X_tune[non_noise_mask], labels[non_noise_mask],
                                   sample_size=3000, random_state=42)
        else:
            sil = np.nan

        results.append({
            'eps': eps,
            'minPts': minpts,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'pct_noise': round(pct_noise, 1),
            'silhouette': round(sil, 4) if not np.isnan(sil) else np.nan
        })
        sil_str = f"{sil:.4f}" if not np.isnan(sil) else "N/A"
        print(f"  eps={eps}, minPts={minpts}: clusters={n_clusters}, "
              f"noise={n_noise}({pct_noise:.1f}%), sil={sil_str}")

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUTPUT_DIR, 'dbscan_tuning_results.csv'), index=False)
print("\nSimpan: dbscan_tuning_results.csv")

# Pilih parameter terbaik: ≥2 cluster, noise < 50%, silhouette tertinggi
df_valid = df_results[
    (df_results['n_clusters'] >= 2) &
    (df_results['pct_noise'] < 50) &
    (df_results['silhouette'].notna())
]

if len(df_valid) > 0:
    best_row    = df_valid.loc[df_valid['silhouette'].idxmax()]
    best_eps    = best_row['eps']
    best_minpts = int(best_row['minPts'])
    print(f"\n>> Parameter Terpilih: eps={best_eps}, minPts={best_minpts}")
    print(f"   Silhouette={best_row['silhouette']:.4f}, "
          f"Clusters={int(best_row['n_clusters'])}, "
          f"Noise={int(best_row['n_noise'])} ({best_row['pct_noise']}%)")
else:
    best_eps    = 0.6
    best_minpts = 8
    print(f"\n>> Fallback: eps={best_eps}, minPts={best_minpts}")


# ==============================================================================
# 4.4 Hasil DBSCAN dengan Parameter Terpilih (seluruh dataset)
# ==============================================================================
print(f"\n=== 4.4 Hasil DBSCAN pada Seluruh Data (eps={best_eps}, minPts={best_minpts}) ===")

db_final     = DBSCAN(eps=best_eps, min_samples=best_minpts, metric='euclidean')
labels_final = db_final.fit_predict(X_scaled)

n_clusters_final = len(set(labels_final)) - (1 if -1 in labels_final else 0)
n_noise_final    = np.sum(labels_final == -1)
pct_noise_final  = n_noise_final / len(labels_final) * 100

print(f"Jumlah Cluster      : {n_clusters_final}")
print(f"Jumlah Noise/Outlier: {n_noise_final} ({pct_noise_final:.1f}%)")
print("\nDistribusi Label Cluster:")
unique, counts = np.unique(labels_final, return_counts=True)
for u, c in zip(unique, counts):
    lbl_str = f"Cluster {u}" if u != -1 else "Noise (-1)"
    print(f"  {lbl_str}: {c} data")

# Silhouette Score
if n_clusters_final >= 2:
    non_noise_mask = labels_final != -1
    sil_final = silhouette_score(X_scaled[non_noise_mask], labels_final[non_noise_mask],
                                 sample_size=10000, random_state=42)
    print(f"\nSilhouette Score (tanpa noise, sample 10k): {sil_final:.4f}")
else:
    sil_final = np.nan
    print("\nSilhouette Score: tidak bisa dihitung (cluster < 2)")

df['dbscan_cluster'] = labels_final


# ==============================================================================
# Visualisasi Hasil DBSCAN
# ==============================================================================
print("\n=== Menyimpan Visualisasi ===")

# PCA untuk proyeksi 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_ratio = pca.explained_variance_ratio_

# Warna cluster
unique_labels = sorted(set(labels_final))
n_real_clusters = len([l for l in unique_labels if l != -1])
palette = plt.cm.Set1(np.linspace(0, 1, max(n_real_clusters, 1)))
color_map = {}
ci = 0
for lbl in unique_labels:
    color_map[lbl] = 'lightgray' if lbl == -1 else palette[ci]
    if lbl != -1:
        ci += 1

fig, axes = plt.subplots(1, 2, figsize=(17, 6))

# Plot 1: PCA 2D
ax = axes[0]
for lbl in unique_labels:
    mask = labels_final == lbl
    n_pts = np.sum(mask)
    lbl_name = f'Cluster {lbl} (n={n_pts})' if lbl != -1 else f'Noise (n={n_pts})'
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=[color_map[lbl]], s=10 if lbl != -1 else 5,
        alpha=0.6 if lbl != -1 else 0.2,
        label=lbl_name
    )
ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)')
ax.set_title(f'DBSCAN — PCA 2D\neps={best_eps}, minPts={best_minpts}, '
             f'{n_clusters_final} cluster, noise={pct_noise_final:.1f}%')
ax.legend(fontsize=7, loc='upper right')

# Plot 2: age vs hours-per-week
ax2 = axes[1]
for lbl in unique_labels:
    mask = labels_final == lbl
    lbl_name = f'Cluster {lbl}' if lbl != -1 else 'Noise'
    ax2.scatter(
        df.loc[mask, 'age'], df.loc[mask, 'hours-per-week'],
        c=[color_map[lbl]], s=8 if lbl != -1 else 4,
        alpha=0.6 if lbl != -1 else 0.2,
        label=lbl_name
    )
ax2.set_xlabel('Age')
ax2.set_ylabel('Hours per Week')
ax2.set_title('DBSCAN — Age vs Hours per Week')
ax2.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dbscan_hasil_clustering.png'), dpi=150)
print("Simpan: dbscan_hasil_clustering.png")

# Heatmap tuning
pivot_sil = df_results.pivot(index='minPts', columns='eps', values='silhouette')
plt.figure(figsize=(10, 5))
sns.heatmap(pivot_sil, annot=True, fmt='.3f', cmap='YlGnBu',
            linewidths=0.5, cbar_kws={'label': 'Silhouette Score'})
plt.title('Heatmap Silhouette Score — Tuning eps & minPts (Adult Incomes)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dbscan_heatmap_tuning.png'), dpi=150)
print("Simpan: dbscan_heatmap_tuning.png")

# Simpan ringkasan evaluasi
eval_path = os.path.join(OUTPUT_DIR, 'dbscan_evaluasi.txt')
with open(eval_path, 'w') as f:
    f.write("=== RINGKASAN EVALUASI DBSCAN (ADULT INCOMES) ===\n")
    f.write(f"Dataset       : adult-incomes-dataset.csv\n")
    f.write(f"Fitur         : {', '.join(fitur)}\n")
    f.write(f"Jumlah Data   : {len(df)}\n")
    f.write(f"eps           : {best_eps}\n")
    f.write(f"minPts        : {best_minpts}\n")
    f.write(f"Jumlah Cluster: {n_clusters_final}\n")
    f.write(f"Jumlah Noise  : {n_noise_final} ({pct_noise_final:.1f}%)\n")
    f.write(f"Silhouette    : {sil_final if not np.isnan(sil_final) else 'N/A'}\n\n")
    f.write("Tabel Tuning:\n")
    f.write(df_results.to_string(index=False))

print(f"Simpan: {eval_path}")
print("\n=== SELESAI: DBSCAN Adult Incomes ===")
