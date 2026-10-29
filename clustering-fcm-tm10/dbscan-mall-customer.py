# ==============================================================================
# PRAKTIKUM MACHINE LEARNING - CLUSTERING MENGGUNAKAN DBSCAN
# Dataset: Mall_Customers.csv
# Anggota C (Raka) — TM10
# ==============================================================================

# ==============================================================================
# 4.1 Persiapan Data (sama seperti FCM, agar konsisten)
# ==============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend untuk simpan gambar
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output-visualisasi-dan-eksplorasi-mall-customer')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
dataset_path = os.path.join(BASE_DIR, 'Mall_Customers.csv')
df = pd.read_csv(dataset_path)
print("=== DBSCAN — Mall Customers Dataset ===")
print("Shape:", df.shape)
print(df.head())

# Fitur yang sama dengan FCM: Annual Income & Spending Score
fitur = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[fitur].values

# Standarisasi (wajib untuk DBSCAN karena berbasis jarak Euclidean)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n=== Statistik Fitur Setelah Standarisasi ===")
print(pd.DataFrame(X_scaled, columns=fitur).describe().round(3))


# ==============================================================================
# 4.3 Penentuan Parameter DBSCAN — k-Distance Graph
# ==============================================================================
print("\n=== 4.3 Penentuan Parameter DBSCAN ===")

# k-Distance Graph digunakan untuk menentukan nilai eps optimal.
# Prinsip: hitung jarak ke-k tetangga terdekat setiap titik, urutkan dari besar ke kecil.
# Titik "siku" (knee/elbow) pada grafik menunjukkan nilai eps yang tepat.
# MinPts standar = 2 * n_fitur = 2 * 2 = 4 → kita uji k=4

k_neighbors = 4

nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean').fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)

# Ambil jarak ke tetangga ke-k (kolom terakhir), lalu urutkan dari besar ke kecil
k_distances = distances[:, k_neighbors - 1]
k_distances_sorted = np.sort(k_distances)[::-1]

plt.figure(figsize=(10, 5))
plt.plot(k_distances_sorted, color='steelblue', linewidth=2)
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='eps = 0.5')
plt.axhline(y=0.6, color='orange', linestyle='--', linewidth=1.5, label='eps = 0.6')
plt.xlabel('Data Points (diurutkan)')
plt.ylabel(f'Jarak ke-{k_neighbors} Tetangga Terdekat')
plt.title(f'k-Distance Graph (k={k_neighbors}) — Mall Customers')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dbscan_kdistance_graph.png'), dpi=150)
print("Simpan: dbscan_kdistance_graph.png")


# ==============================================================================
# Tuning eps dan minPts
# ==============================================================================
print("\n=== Tuning eps dan minPts ===")

# Grid search kombinasi eps dan min_samples
eps_values    = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
minpts_values = [3, 4, 5, 6, 8]

results = []
for eps in eps_values:
    for minpts in minpts_values:
        db = DBSCAN(eps=eps, min_samples=minpts, metric='euclidean')
        labels = db.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = np.sum(labels == -1)
        pct_noise  = n_noise / len(labels) * 100

        # Silhouette hanya bisa dihitung jika ada ≥2 cluster dan bukan semua noise
        if n_clusters >= 2 and n_noise < len(labels):
            non_noise_mask = labels != -1
            sil = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
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

df_results = pd.DataFrame(results)
print("\nTabel Tuning eps & minPts:")
print(df_results.to_string(index=False))

# Simpan tabel tuning ke CSV
df_results.to_csv(os.path.join(OUTPUT_DIR, 'dbscan_tuning_results.csv'), index=False)
print("Simpan: dbscan_tuning_results.csv")

# Pilih parameter terbaik: cluster ≥ 2, noise < 30%, silhouette maksimum
df_valid = df_results[
    (df_results['n_clusters'] >= 2) &
    (df_results['pct_noise'] < 30) &
    (df_results['silhouette'].notna())
]

if len(df_valid) > 0:
    best_row = df_valid.loc[df_valid['silhouette'].idxmax()]
    best_eps    = best_row['eps']
    best_minpts = int(best_row['minPts'])
    print(f"\n>> Parameter Terpilih: eps={best_eps}, minPts={best_minpts}")
    print(f"   Silhouette={best_row['silhouette']:.4f}, Clusters={int(best_row['n_clusters'])}, "
          f"Noise={int(best_row['n_noise'])} ({best_row['pct_noise']}%)")
else:
    # Fallback ke nilai yang menghasilkan cluster terbanyak
    best_eps    = 0.5
    best_minpts = 4
    print(f"\n>> Fallback parameter: eps={best_eps}, minPts={best_minpts}")


# ==============================================================================
# 4.4 Hasil DBSCAN dengan Parameter Terpilih
# ==============================================================================
print(f"\n=== 4.4 Hasil DBSCAN (eps={best_eps}, minPts={best_minpts}) ===")

db_final = DBSCAN(eps=best_eps, min_samples=best_minpts, metric='euclidean')
labels_final = db_final.fit_predict(X_scaled)

n_clusters_final = len(set(labels_final)) - (1 if -1 in labels_final else 0)
n_noise_final    = np.sum(labels_final == -1)
pct_noise_final  = n_noise_final / len(labels_final) * 100

print(f"Jumlah Cluster      : {n_clusters_final}")
print(f"Jumlah Noise/Outlier: {n_noise_final} ({pct_noise_final:.1f}%)")
print("\nDistribusi Label Cluster:")
unique, counts = np.unique(labels_final, return_counts=True)
for u, c in zip(unique, counts):
    label_str = f"Cluster {u}" if u != -1 else "Noise (-1)"
    print(f"  {label_str}: {c} data")

# Silhouette Score final (tanpa noise)
if n_clusters_final >= 2:
    non_noise_mask = labels_final != -1
    sil_final = silhouette_score(X_scaled[non_noise_mask], labels_final[non_noise_mask])
    print(f"\nSilhouette Score (tanpa noise): {sil_final:.4f}")
else:
    sil_final = np.nan
    print("\nSilhouette Score: tidak bisa dihitung (cluster < 2)")

# Tambahkan label ke dataframe
df['dbscan_cluster'] = labels_final


# ==============================================================================
# Visualisasi Hasil DBSCAN
# ==============================================================================
print("\n=== Menyimpan Visualisasi Hasil DBSCAN ===")

# Siapkan warna: noise berwarna abu-abu, cluster berwarna berbeda
unique_labels = sorted(set(labels_final))
palette = plt.cm.Set1(np.linspace(0, 1, max(n_clusters_final, 1)))
color_map = {}
cluster_idx = 0
for lbl in unique_labels:
    if lbl == -1:
        color_map[lbl] = 'lightgray'
    else:
        color_map[lbl] = palette[cluster_idx]
        cluster_idx += 1

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Scatter pada skala asli
ax = axes[0]
for lbl in unique_labels:
    mask = labels_final == lbl
    label_name = f'Cluster {lbl}' if lbl != -1 else f'Noise ({np.sum(mask)})'
    ax.scatter(
        df.loc[mask, 'Annual Income (k$)'],
        df.loc[mask, 'Spending Score (1-100)'],
        c=[color_map[lbl]],
        s=80 if lbl != -1 else 40,
        alpha=0.8 if lbl != -1 else 0.4,
        edgecolors='k' if lbl != -1 else 'none',
        linewidths=0.5,
        label=label_name
    )
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.set_title(f'DBSCAN (eps={best_eps}, minPts={best_minpts})\n'
             f'{n_clusters_final} cluster, {n_noise_final} noise ({pct_noise_final:.1f}%)')
ax.legend(loc='upper left', fontsize=8)

# Plot 2: Scatter pada skala standardisasi
ax2 = axes[1]
for lbl in unique_labels:
    mask = labels_final == lbl
    label_name = f'Cluster {lbl}' if lbl != -1 else f'Noise ({np.sum(mask)})'
    ax2.scatter(
        X_scaled[mask, 0], X_scaled[mask, 1],
        c=[color_map[lbl]],
        s=80 if lbl != -1 else 40,
        alpha=0.8 if lbl != -1 else 0.4,
        edgecolors='k' if lbl != -1 else 'none',
        linewidths=0.5,
        label=label_name
    )
ax2.set_xlabel('Annual Income (scaled)')
ax2.set_ylabel('Spending Score (scaled)')
ax2.set_title('DBSCAN — Skala Terstandarisasi')
ax2.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dbscan_hasil_clustering.png'), dpi=150)
print("Simpan: dbscan_hasil_clustering.png")

# Heatmap tuning: Silhouette Score per kombinasi eps x minPts
pivot_sil = df_results.pivot(index='minPts', columns='eps', values='silhouette')
plt.figure(figsize=(9, 5))
sns.heatmap(pivot_sil, annot=True, fmt='.3f', cmap='YlGnBu',
            linewidths=0.5, cbar_kws={'label': 'Silhouette Score'})
plt.title('Heatmap Silhouette Score — Tuning eps & minPts (Mall Customers)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dbscan_heatmap_tuning.png'), dpi=150)
print("Simpan: dbscan_heatmap_tuning.png")

# Simpan ringkasan evaluasi
eval_path = os.path.join(OUTPUT_DIR, 'dbscan_evaluasi.txt')
with open(eval_path, 'w') as f:
    f.write("=== RINGKASAN EVALUASI DBSCAN (MALL CUSTOMERS) ===\n")
    f.write(f"Dataset       : Mall_Customers.csv\n")
    f.write(f"Fitur         : {', '.join(fitur)}\n")
    f.write(f"Jumlah Data   : {len(df)}\n")
    f.write(f"eps           : {best_eps}\n")
    f.write(f"minPts        : {best_minpts}\n")
    f.write(f"Jumlah Cluster: {n_clusters_final}\n")
    f.write(f"Jumlah Noise  : {n_noise_final} ({pct_noise_final:.1f}%)\n")
    f.write(f"Silhouette    : {sil_final:.4f}\n\n")
    f.write("Tabel Tuning:\n")
    f.write(df_results.to_string(index=False))

print(f"Simpan: {eval_path}")
print("\n=== SELESAI: DBSCAN Mall Customers ===")
