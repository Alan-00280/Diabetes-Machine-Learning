"""
TM10 - K-Modes untuk Dataset Mall Customer
Anggota A: Vito (K-Modes)

Fitur: Annual Income (k$) dan Spending Score (1-100),
       disamakan dengan FCM (Alan) dan DBSCAN (Raka) agar hasilnya sebanding.

Tahapan:
  1. Binning fitur numerik menjadi 5 tingkat kategori
  2. Elbow Method untuk menentukan jumlah cluster
  3. Membangun model K-Modes
  4. Silhouette Score dengan jarak Hamming
  5. Uji kestabilan antar seed (Adjusted Rand Index)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kmodes.kmodes import KModes
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (adjusted_rand_score, silhouette_samples,
                             silhouette_score)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SETUP PATH DAN FOLDER OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, 'Mall_Customers.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output-visualisasi-dan-eksplorasi-mall-customer')
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATASET)

print("=" * 70)
print("  K-MODES - MALL CUSTOMER")
print("=" * 70)
print(f"\nJumlah data : {df.shape[0]} baris")
print(f"Kolom       : {df.columns.tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. BINNING FITUR NUMERIK
# K-Modes hanya dapat memproses data kategorik, sedangkan kedua fitur pada
# Mall Customer bersifat numerik. Karena itu keduanya dikategorikan menjadi
# empat tingkat berdasarkan kuantil.
# ─────────────────────────────────────────────────────────────────────────────
TINGKAT = ['Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']

print("\n" + "=" * 70)
print("1. BINNING FITUR NUMERIK MENJADI KATEGORI")
print("=" * 70)

for kolom in ['Annual Income (k$)', 'Spending Score (1-100)']:
    batas = df[kolom].quantile([0, 0.25, 0.5, 0.75, 1.0]).round(1).tolist()
    print(f"\n{kolom}")
    print(f"  Batas kuantil: {batas}")

df['Income_Kategori'] = pd.qcut(df['Annual Income (k$)'], q=4,
                                labels=TINGKAT, duplicates='drop')
df['Spending_Kategori'] = pd.qcut(df['Spending Score (1-100)'], q=4,
                                  labels=TINGKAT, duplicates='drop')

for kolom in ['Income_Kategori', 'Spending_Kategori']:
    print(f"\nDistribusi {kolom}:")
    print(df[kolom].value_counts().reindex(TINGKAT).to_string())

FITUR = ['Income_Kategori', 'Spending_Kategori']
X = df[FITUR].astype(str).values
print(f"\nJumlah kombinasi kategori unik: {len(pd.DataFrame(X).drop_duplicates())}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ELBOW METHOD
# Semakin banyak cluster, cost semakin kecil. Titik ketika penurunan mulai
# melandap dianggap sebagai jumlah cluster yang optimal.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. ELBOW METHOD")
print("=" * 70)

DAFTAR_K = list(range(1, 9))
cost = []
for k in DAFTAR_K:
    km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=0, random_state=111)
    km.fit(X)
    cost.append(km.cost_)

print("\n   k  |    Cost    | Penurunan")
print("  ----+------------+----------")
for i, k in enumerate(DAFTAR_K):
    penurunan = '-' if i == 0 else f"{cost[i - 1] - cost[i]:.2f}"
    print(f"  {k:>3}  | {cost[i]:>9.2f}  | {penurunan}")

pd.DataFrame({'k': DAFTAR_K, 'cost': cost}).to_csv(
    os.path.join(OUTPUT_DIR, 'kmodes_elbow_cost.csv'), index=False)

# Jumlah cluster terpilih berdasarkan titik siku pada Elbow Method, yaitu k
# terakhir sebelum penurunan cost melandap, diperkuat oleh Silhouette Score
# tertinggi serta hasil uji kestabilan pada bagian sensitivitas di bawah.
K_TERPILIH = 4

# ─────────────────────────────────────────────────────────────────────────────
# 4. MEMBANGUN MODEL K-MODES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"3. MEMBANGUN MODEL K-MODES (k={K_TERPILIH})")
print("=" * 70)

model = KModes(n_clusters=K_TERPILIH, init='Huang', n_init=5, verbose=0, random_state=111)
label = model.fit_predict(X)
df['cluster'] = label

print(f"\nCost model akhir : {model.cost_:.2f}")
print("\nModus tiap cluster (kategori dominan):")
modus = pd.DataFrame(model.cluster_centroids_, columns=FITUR)
modus.index.name = 'cluster'
print(modus.to_string())

# Rata-rata nilai asli tiap cluster untuk membantu interpretasi
print("\nRata-rata nilai asli per cluster:")
rata_rata = df.groupby('cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean().round(1)
print(rata_rata.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 5. SILHOUETTE SCORE DENGAN JARAK HAMMING
# Untuk data kategorik, kemiripan diukur dengan proporsi atribut yang berbeda.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. SILHOUETTE SCORE (JARAK HAMMING)")
print("=" * 70)


def matriks_jarak_hamming(data_kategorikal):
    """Jarak Hamming: proporsi fitur yang nilainya berbeda antara dua data."""
    kode = np.column_stack([
        pd.factorize(data_kategorikal[:, j])[0]
        for j in range(data_kategorikal.shape[1])
    ])
    return squareform(pdist(kode, metric='hamming'))


jarak_hamming = matriks_jarak_hamming(X)
silhouette_hamming = silhouette_score(jarak_hamming, label, metric='precomputed')
silhouette_tiap_data = silhouette_samples(jarak_hamming, label, metric='precomputed')

print(f"\nSilhouette Score keseluruhan : {silhouette_hamming:.4f}")
print("\nSilhouette Score tiap cluster:")
for cluster in sorted(set(label)):
    jumlah = int((label == cluster).sum())
    sil = silhouette_tiap_data[label == cluster].mean()
    print(f"  Cluster {cluster}: {jumlah:>3} anggota | silhouette={sil:.4f}")

# Pembanding: jarak Euclidean atas hasil label-encoding (kurang tepat untuk kategorik)
X_encoded = pd.DataFrame(X).apply(lambda s: pd.factorize(s)[0]).values
silhouette_euclid = silhouette_score(X_encoded, label, metric='euclidean')
print(f"\nPembanding - jarak Euclidean atas label-encoding : {silhouette_euclid:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. UJI KESTABILAN ANTAR SEED
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. UJI KESTABILAN ANTAR SEED")
print("=" * 70)

daftar_seed = [111, 7, 42, 2024]
label_per_seed = {}
for seed in daftar_seed:
    km_seed = KModes(n_clusters=K_TERPILIH, init='Huang', n_init=5, verbose=0, random_state=seed)
    label_per_seed[seed] = km_seed.fit_predict(X)
    print(f"  seed={seed:<5} cost={km_seed.cost_:>8.2f}")

ari = np.zeros((len(daftar_seed), len(daftar_seed)))
for a in range(len(daftar_seed)):
    for b in range(len(daftar_seed)):
        ari[a, b] = adjusted_rand_score(label_per_seed[daftar_seed[a]],
                                        label_per_seed[daftar_seed[b]])

print("\nAdjusted Rand Index antar seed (1,0000 berarti identik):")
print(pd.DataFrame(ari, index=daftar_seed, columns=daftar_seed).round(4).to_string())
rata_ari = ari[np.triu_indices(len(daftar_seed), 1)].mean()
print(f"\nRata-rata ARI antar seed berbeda: {rata_ari:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6b. UJI SENSITIVITAS TERHADAP BINNING DAN JUMLAH CLUSTER
# Membandingkan beberapa kombinasi binning dan k, supaya pemilihan model
# didukung bukti dan bukan tebakan.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. UJI SENSITIVITAS BINNING DAN JUMLAH CLUSTER")
print("=" * 70)

data_binning = {}
label_binning = {
    4: ['Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'],
    5: ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'],
}
for jumlah_binning in [4, 5]:
    inc = pd.qcut(df['Annual Income (k$)'], q=jumlah_binning,
                  labels=label_binning[jumlah_binning], duplicates='drop')
    spn = pd.qcut(df['Spending Score (1-100)'], q=jumlah_binning,
                  labels=label_binning[jumlah_binning], duplicates='drop')
    data_binning[jumlah_binning] = np.column_stack([inc.astype(str), spn.astype(str)])

hasil_sensitivitas = []
for jumlah_binning in [4, 5]:
    X_uji = data_binning[jumlah_binning]
    kode_uji = np.column_stack([pd.factorize(X_uji[:, j])[0] for j in range(2)])
    jarak_uji = squareform(pdist(kode_uji, metric='hamming'))
    for k_uji in [3, 4, 5]:
        km_uji = KModes(n_clusters=k_uji, init='Huang', n_init=5, verbose=0, random_state=111)
        pred_uji = km_uji.fit_predict(X_uji)
        sil_uji = silhouette_score(jarak_uji, pred_uji, metric='precomputed')

        label_uji = []
        for seed in [111, 7, 42, 2024]:
            km_seed = KModes(n_clusters=k_uji, init='Huang', n_init=5,
                             verbose=0, random_state=seed)
            label_uji.append(km_seed.fit_predict(X_uji))
        ari_uji = np.mean([
            adjusted_rand_score(label_uji[i], label_uji[j])
            for i in range(len(label_uji)) for j in range(i + 1, len(label_uji))
        ])

        hasil_sensitivitas.append({
            'binning': jumlah_binning,
            'k': k_uji,
            'cost': round(km_uji.cost_, 2),
            'silhouette': round(sil_uji, 4),
            'rata_ari': round(ari_uji, 4),
        })

df_sensitivitas = pd.DataFrame(hasil_sensitivitas)
print()
print(df_sensitivitas.to_string(index=False))
df_sensitivitas.to_csv(os.path.join(OUTPUT_DIR, 'kmodes_sensitivitas.csv'), index=False)
print("\n[OUTPUT] kmodes_sensitivitas.csv tersimpan")

baris_terbaik = df_sensitivitas.loc[df_sensitivitas['silhouette'].idxmax()]
print(f"\nKombinasi terbaik: binning {int(baris_terbaik['binning'])} tingkat dengan "
      f"k={int(baris_terbaik['k'])} "
      f"(silhouette={baris_terbaik['silhouette']}, ARI={baris_terbaik['rata_ari']})")

# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISASI
# ─────────────────────────────────────────────────────────────────────────────
# Grafik 1: Elbow Method
plt.figure(figsize=(9, 5))
plt.plot(DAFTAR_K, cost, 'o-', linewidth=2, markersize=8, color='steelblue')
plt.axvline(K_TERPILIH, color='red', linestyle='--',
            label=f'k terpilih = {K_TERPILIH}')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Cost')
plt.title('Elbow Method - K-Modes (Mall Customer)')
plt.xticks(DAFTAR_K)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'kmodes_elbow_cost.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n[OUTPUT] kmodes_elbow_cost.png tersimpan")

# Grafik 2: Hasil cluster pada fitur asli
warna = ['#4C72B0', '#55A868', '#C44E52', '#DD8452', '#8172B2', '#937860', '#64B5CD']
plt.figure(figsize=(10, 7))
for cluster in sorted(set(label)):
    mask = label == cluster
    plt.scatter(df.loc[mask, 'Annual Income (k$)'],
                df.loc[mask, 'Spending Score (1-100)'],
                s=90, alpha=0.7, color=warna[cluster % len(warna)],
                label=f'Cluster {cluster}')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title(f'Hasil Clustering K-Modes (k={K_TERPILIH}) - Mall Customer')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'kmodes_hasil_cluster.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OUTPUT] kmodes_hasil_cluster.png tersimpan")

# Grafik 3: Silhouette tiap cluster
plt.figure(figsize=(8, 5))
daftar_cluster = sorted(set(label))
nilai_sil = [silhouette_tiap_data[label == c].mean() for c in daftar_cluster]
plt.bar([f'Cluster {c}' for c in daftar_cluster], nilai_sil, color='#4C72B0')
plt.axhline(silhouette_hamming, color='red', linestyle='--',
            label=f'Rata-rata = {silhouette_hamming:.4f}')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Tiap Cluster (Jarak Hamming)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'kmodes_silhouette_cluster.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OUTPUT] kmodes_silhouette_cluster.png tersimpan")

# Grafik 4: Kestabilan antar seed
plt.figure(figsize=(7, 6))
sns.heatmap(ari, annot=True, fmt='.3f', cmap='rocket_r', vmin=0, vmax=1,
            xticklabels=daftar_seed, yticklabels=daftar_seed,
            cbar_kws={'label': 'Adjusted Rand Index'})
plt.xlabel('Seed')
plt.ylabel('Seed')
plt.title('Kestabilan Cluster Antar Seed')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'kmodes_kestabilan_ari.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OUTPUT] kmodes_kestabilan_ari.png tersimpan")

# ─────────────────────────────────────────────────────────────────────────────
# 8. MENYIMPAN RINGKASAN
# ─────────────────────────────────────────────────────────────────────────────
ringkasan = []
for cluster in sorted(set(label)):
    mask = label == cluster
    ringkasan.append({
        'cluster': cluster,
        'jumlah_anggota': int(mask.sum()),
        'persen_anggota': round(mask.sum() / len(label) * 100, 2),
        'silhouette': round(silhouette_tiap_data[mask].mean(), 4),
        'modus_income': model.cluster_centroids_[cluster][0],
        'modus_spending': model.cluster_centroids_[cluster][1],
        'rata_income': round(df.loc[mask, 'Annual Income (k$)'].mean(), 1),
        'rata_spending': round(df.loc[mask, 'Spending Score (1-100)'].mean(), 1),
    })
df_ringkasan = pd.DataFrame(ringkasan)
df_ringkasan.to_csv(os.path.join(OUTPUT_DIR, 'kmodes_ringkasan_cluster.csv'), index=False)
print("\n[OUTPUT] kmodes_ringkasan_cluster.csv tersimpan")

with open(os.path.join(OUTPUT_DIR, 'evaluasi_model_kmodes.txt'), 'w') as f:
    f.write("=== RINGKASAN EVALUASI CLUSTERING K-MODES (TM10) ===\n")
    f.write("Dataset: Mall_Customers.csv\n")
    f.write("Fitur: Annual Income (k$), Spending Score (1-100)\n")
    f.write(f"Jumlah Baris: {len(df)}\n")
    f.write(f"Jumlah Cluster (k): {K_TERPILIH}\n")
    f.write("Binning: kuantil 4 tingkat (Rendah, Sedang, Tinggi, Sangat Tinggi)\n")
    f.write(f"Cost akhir: {model.cost_:.2f}\n")
    f.write(f"Silhouette Score (Hamming): {silhouette_hamming:.4f}\n")
    f.write(f"Silhouette Score (Euclidean atas label-encoding): {silhouette_euclid:.4f}\n")
    f.write(f"Rata-rata ARI antar seed: {rata_ari:.4f}\n\n")
    f.write("Uji Sensitivitas Binning dan Jumlah Cluster:\n")
    f.write(df_sensitivitas.to_string(index=False) + "\n\n")
    f.write("Modus Tiap Cluster:\n")
    f.write(modus.to_string() + "\n\n")
    f.write("Rata-rata Nilai Asli per Cluster:\n")
    f.write(rata_rata.to_string() + "\n")
print("[OUTPUT] evaluasi_model_kmodes.txt tersimpan")

print("\n=== SELESAI ===")
print(f"Semua output tersimpan di: {OUTPUT_DIR}")
