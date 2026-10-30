# ==============================================================================
# PRAKTIKUM MACHINE LEARNING - CLUSTERING MENGGUNAKAN K-MODES
# Dataset: adult-incomes-dataset.csv
# ==============================================================================

# 4.1 Eksplorasi Data
# 1 # import tools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# Direktori kerja dan output untuk menyimpan gambar visualisasi dan eksplorasi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output-visualisasi-dan-eksplorasi')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2 # import data
dataset_path = os.path.join(BASE_DIR, 'adult-incomes-dataset.csv')
if not os.path.exists(dataset_path):
    dataset_path = 'adult-incomes-dataset.csv'
df = pd.read_csv(dataset_path)
print("5 Data Teratas:")
print(df.head())

# 3 # amati bentuk data
print("\nDimensi Data (Baris, Kolom):")
print(df.shape)

# 4 # Melihat ringkasan statistik deskriptif dari DataFrame
print("\nRingkasan Statistik Deskriptif:")
print(df.describe())

# 5 # cek dan tangani null data
print("\n=== Pengecekan Missing Values ===")
print("Jumlah Missing Values (NaN):")
print(df.isnull().sum())
print("\nJumlah Missing Values Karakter Khusus '?':")
print((df == '?').sum())

# Penanganan: Ganti karakter '?' dengan NaN, lalu imputasi kolom berkategori dengan nilai modus
df = df.replace('?', np.nan)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nCek Ulang Missing Values Setelah Penanganan:")
print(df.isnull().sum())

# 6 # cek dan tangani outlier
numeric_cols = df.select_dtypes(include=[np.number]).columns
print("\n=== Deteksi Outlier (Metode IQR) Sebelum Penanganan ===")
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outlier_count = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
    print(f"- {col}: {outlier_count} outlier")

# Boxplot sebelum penanganan outlier
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_cols])
plt.title('Boxplot Fitur Numerik Sebelum Penanganan Outlier')
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_DIR, '4.1_boxplot_outlier_sebelum.png'), bbox_inches='tight')
plt.show()
plt.close()

# Penanganan outlier: Capping (Winsorization) dengan batas IQR agar tidak menghilangkan data
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# Boxplot setelah penanganan outlier
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_cols])
plt.title('Boxplot Fitur Numerik Setelah Capping Outlier')
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_DIR, '4.1_boxplot_outlier_setelah_capping.png'), bbox_inches='tight')
plt.show()
plt.close()

# 6.1 # Kategorisasi sederhana kolom numerik untuk K-Modes Clustering
# Karena K-Modes bekerja secara spesifik pada data kategorikal, fitur numerik diubah ke kategori (binning)
print("\n=== Kategorisasi Kolom Numerik untuk K-Modes ===")

# Kategorisasi Umur (age)
df['age'] = pd.cut(
    df['age'],
    bins=[0, 25, 45, 65, 100],
    labels=['Muda', 'Dewasa', 'Paruh-Baya', 'Lansia'],
    right=True
)

# Kategorisasi Jam Kerja per Minggu (hours-per-week)
df['hours-per-week'] = pd.cut(
    df['hours-per-week'],
    bins=[0, 39, 40, 100],
    labels=['Part-time', 'Full-time', 'Over-time'],
    right=True
)

# Kategorisasi Capital Gain & Capital Loss
df['capital-gain'] = pd.cut(
    df['capital-gain'],
    bins=[-1, 0, np.inf],
    labels=['No Gain', 'Gain'],
    right=True
)

df['capital-loss'] = pd.cut(
    df['capital-loss'],
    bins=[-1, 0, np.inf],
    labels=['No Loss', 'Loss'],
    right=True
)

# Kategorisasi Tingkat Pendidikan Numerik (educational-num)
df['educational-num'] = pd.cut(
    df['educational-num'],
    bins=[0, 8, 12, 16],
    labels=['Pendidikan Dasar', 'Pendidikan Menengah', 'Pendidikan Tinggi'],
    right=True
)

# Kategorisasi Bobot Sampling (fnlwgt) berdasarkan kuantil
df['fnlwgt'] = pd.qcut(
    df['fnlwgt'],
    q=3,
    labels=['Low Weight', 'Medium Weight', 'High Weight']
)

# Mengubah semua kolom numerik yang telah dibin menjadi tipe string/kategori
for col in numeric_cols:
    df[col] = df[col].astype(str)

print("Tipe data setelah kategorisasi:")
print(df.dtypes)
print("\n5 Data Teratas Setelah Pembersihan dan Kategorisasi:")
print(df.head())


# 4.2 Visualisasi Data
# 7 # amati bentuk visual masing-masing fitur
plt.style.use('fivethirtyeight')
plt.figure(figsize = (15 , 6))
n = 0
for x in ['workclass', 'education', 'marital-status']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.countplot(y = df[x])
    plt.title('Countplot of {}'.format(x))
plt.savefig(os.path.join(OUTPUT_DIR, '4.2_countplot_fitur_kategorikal.png'), bbox_inches='tight')
plt.show()
plt.close()

# 8 # Ploting untuk mencari relasi antara fitur kategorikal
plt.figure(figsize = (15 , 10))
n = 0
for x in ['education', 'occupation']:
    n += 1
    plt.subplot(2 , 1 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.countplot(x = x, hue = 'gender', data = df)
    plt.xticks(rotation = 45)
    plt.title('Distribution of {} by Gender'.format(x))
plt.savefig(os.path.join(OUTPUT_DIR, '4.2_relasi_kategorikal_by_gender.png'), bbox_inches='tight')
plt.show()
plt.close()

# 9 # Melihat sebaran fitur kategorikal pada income
plt.figure(figsize = (15 , 8))
sns.countplot(x = 'workclass', hue = 'income', data = df)
plt.title('Workclass vs Income')
plt.xticks(rotation = 45)
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, '4.2_workclass_vs_income.png'), bbox_inches='tight')
plt.show()
plt.close()

# 9.1 # Visualisasi sebaran fitur hasil kategorisasi terhadap income
plt.figure(figsize = (15 , 5))
plt.subplot(1 , 2 , 1)
sns.countplot(x = 'age', hue = 'income', data = df)
plt.title('Age Group vs Income')

plt.subplot(1 , 2 , 2)
sns.countplot(x = 'hours-per-week', hue = 'income', data = df)
plt.title('Hours per Week Group vs Income')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4.2_kategorisasi_numerik_vs_income.png'), bbox_inches='tight')
plt.show()
plt.close()


# 4.3 Penentuan Jumlah Cluster
# 10 # Merancang K-Modes
# Menentukan nilai k yang sesuai dengan Elbow-Method (Cost)
# Memilih fitur-fitur kategorikal untuk clustering K-Modes
X1 = df[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender']].astype(str).values

cost = []
for n in range(1 , 6):
    algorithm = KModes(n_clusters = n, init = 'Huang', n_init = 5, verbose = 0, random_state = 111)
    algorithm.fit(X1)
    cost.append(algorithm.cost_)

# Plot bentuk visual elbow
plt.figure(1 , figsize = (15 , 6))
plt.plot(range(1 , 6) , cost , 'o')
plt.plot(range(1 , 6) , cost , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Cost')
plt.title('Elbow Method - K-Modes')
plt.xticks(range(1, 6))
plt.savefig(os.path.join(OUTPUT_DIR, '4.3_elbow_kmodes.png'), bbox_inches='tight')
plt.show()
plt.close()


# 4.4 Hasil Clustering
# 11 # Membangun K-Modes
# k=3 dipilih berdasarkan Elbow Method (titik siku paling jelas di k=3)
algorithm = KModes(n_clusters=3, init='Huang', n_init=5, verbose=0, random_state=111)
algorithm.fit(X1)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centroids_

# 12 # Menyiapkan data untuk bentuk visual cluster
# Karena K-Modes bekerja pada data kategorikal berdimensi banyak,
# encode data kategorikal ke angka agar dapat diproyeksikan untuk visualisasi
X1_encoded = pd.DataFrame(X1).apply(LabelEncoder().fit_transform).values

# 13 # Melihat bentuk visual cluster
plt.figure(1 , figsize = (15 , 7))
plt.clf()
plt.scatter(x = X1_encoded[:, 0], y = X1_encoded[:, 1], c = labels2, cmap = plt.cm.Pastel2, s = 200, alpha = 0.6)
plt.ylabel('Feature 2 (Encoded)') , plt.xlabel('Feature 1 (Encoded)')
plt.title('Visualisasi Cluster K-Modes')
plt.colorbar(label='Cluster')
plt.savefig(os.path.join(OUTPUT_DIR, '4.4_scatter_cluster.png'), bbox_inches='tight')
plt.show()
plt.close()


# ==============================================================================
# 4.5 Evaluasi Model
# K-Modes tidak punya "akurasi", jadi kualitas cluster dievaluasi dari beberapa sisi:
#   1) cost tiap nilai k   -> seberapa besar ketidakmiripan di dalam cluster
#   2) Silhouette Score    -> seberapa baik cluster terpisah satu sama lain
#   3) ukuran cluster      -> apakah jumlah anggota antar cluster seimbang
#   4) profil cluster      -> ciri khas (kategori dominan) tiap cluster
#   5) uji kestabilan      -> apakah cluster tetap sama saat seed diubah
# Catatan: K-Modes bekerja pada data kategorikal, sehingga kemiripan antar data
# diukur dengan JARAK HAMMING (proporsi fitur yang nilainya berbeda), bukan
# jarak Euclidean atas hasil label-encoding.
# ==============================================================================

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (adjusted_rand_score, silhouette_samples,
                             silhouette_score)

OUTPUT_EVAL = os.path.join(BASE_DIR, 'output-evaluasi-model')
os.makedirs(OUTPUT_EVAL, exist_ok=True)

FITUR = ['workclass', 'education', 'marital-status', 'occupation',
         'relationship', 'race', 'gender']

print("\n" + "=" * 70)
print("4.5 EVALUASI MODEL K-MODES")
print("=" * 70)

# 14a # Cost tiap nilai k (dari tahap 4.3) dan cost model akhir
print("\n[1] COST TIAP NILAI K (semakin kecil, semakin rapat isi cluster)")
for nilai_k, nilai_cost in zip(range(1, 6), cost):
    tanda = "   <-- dipilih" if nilai_k == 3 else ""
    print(f"    k={nilai_k}: cost={nilai_cost:>12,.2f}{tanda}")

# 14b # Silhouette Score dengan jarak Hamming
# Perhitungan pada seluruh 48.842 baris memerlukan matriks jarak berukuran
# 48.842 x 48.842, yaitu sekitar 9,5 GB, sehingga dipakai sampel acak 5.000 baris.
UKURAN_SAMPEL = 5000


def matriks_jarak_hamming(data_kategorikal):
    """Jarak Hamming: proporsi fitur yang nilainya berbeda antara dua data."""
    kode = np.column_stack([
        pd.factorize(data_kategorikal[:, j])[0]
        for j in range(data_kategorikal.shape[1])
    ])
    return squareform(pdist(kode, metric='hamming'))


rng = np.random.default_rng(42)
idx_sampel = rng.choice(len(X1), size=UKURAN_SAMPEL, replace=False)
X_sampel = X1[idx_sampel]
y_sampel = labels2[idx_sampel]

jarak_hamming = matriks_jarak_hamming(X_sampel)
silhouette_hamming = silhouette_score(jarak_hamming, y_sampel, metric='precomputed')
silhouette_tiap_data = silhouette_samples(jarak_hamming, y_sampel, metric='precomputed')

# Pembanding: cara label-encoding + jarak Euclidean (kurang tepat untuk data kategorikal)
silhouette_euclid = silhouette_score(X1_encoded[idx_sampel], y_sampel, metric='euclidean')

print(f"\n[2] SILHOUETTE SCORE (sampel acak {UKURAN_SAMPEL:,} baris, seed=42)")
print(f"    Jarak Hamming (tepat untuk data kategorikal) : {silhouette_hamming:>7.4f}")
print(f"    Jarak Euclidean atas label-encoding          : {silhouette_euclid:>7.4f}")
print("    Nilai mendekati 1 berarti cluster rapat dan terpisah dengan baik.")

# 14c # Ukuran dan kualitas tiap cluster
print("\n[3] UKURAN DAN KUALITAS TIAP CLUSTER")
for cluster in sorted(set(labels2)):
    jumlah = int((labels2 == cluster).sum())
    sil_cluster = silhouette_tiap_data[y_sampel == cluster].mean()
    print(f"    Cluster {cluster}: {jumlah:>6,} anggota ({jumlah / len(labels2) * 100:5.2f}%)"
          f" | silhouette={sil_cluster:>7.4f}")

# 14d # Profil tiap cluster (kategori dominan dari centroid)
tabel_centroid = pd.DataFrame(centroids2, columns=FITUR)
print("\n[4] PROFIL TIAP CLUSTER (kategori paling dominan)")
print(tabel_centroid.to_string())

tabel_profil = pd.DataFrame(index=FITUR,
                            columns=[f"Cluster {i}" for i in range(3)],
                            dtype=float)
for i in range(3):
    anggota = X1[labels2 == i]
    for j, fitur in enumerate(FITUR):
        tabel_profil.loc[fitur, f"Cluster {i}"] = round(
            (anggota[:, j] == centroids2[i][j]).mean() * 100, 1
        )
print("\n    Persentase anggota yang nilainya sama dengan kategori dominan:")
print(tabel_profil.to_string())

# 14e # Uji kestabilan: jalankan ulang dengan seed berbeda, bandingkan dengan ARI
print("\n[5] UJI KESTABILAN (K-Modes dijalankan ulang dengan seed berbeda)")
daftar_seed = [111, 7, 42, 2024]
label_per_seed = {}
for seed in daftar_seed:
    km_seed = KModes(n_clusters=3, init='Huang', n_init=5, verbose=0, random_state=seed)
    label_per_seed[seed] = km_seed.fit_predict(X1)
    print(f"    seed={seed:<5} cost={km_seed.cost_:>12,.2f}")

print("\n    Adjusted Rand Index antar hasil (1,0000 berarti cluster identik):")
ari = np.zeros((len(daftar_seed), len(daftar_seed)))
for a in range(len(daftar_seed)):
    for b in range(len(daftar_seed)):
        ari[a, b] = adjusted_rand_score(
            label_per_seed[daftar_seed[a]], label_per_seed[daftar_seed[b]]
        )
print(pd.DataFrame(ari, index=daftar_seed, columns=daftar_seed).round(4).to_string())
print(f"\n    Rata-rata ARI antar seed berbeda: {ari[np.triu_indices(len(daftar_seed), 1)].mean():.4f}")

# 14f # Grafik hasil evaluasi
# Grafik 1: silhouette tiap cluster
plt.figure(figsize=(8, 5))
daftar_cluster = sorted(set(labels2))
nilai_sil = [silhouette_tiap_data[y_sampel == c].mean() for c in daftar_cluster]
plt.bar([f'Cluster {c}' for c in daftar_cluster], nilai_sil, color='#4C72B0')
plt.axhline(silhouette_hamming, color='red', linestyle='--',
            label=f'Rata-rata = {silhouette_hamming:.4f}')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Tiap Cluster (Jarak Hamming)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_EVAL, '4.5_silhouette_tiap_cluster.png'), bbox_inches='tight')
plt.close()

# Grafik 2: ukuran tiap cluster
plt.figure(figsize=(8, 5))
jumlah_anggota = [int((labels2 == c).sum()) for c in daftar_cluster]
plt.bar([f'Cluster {c}' for c in daftar_cluster], jumlah_anggota, color='#55A868')
for i, jumlah in enumerate(jumlah_anggota):
    plt.text(i, jumlah, f'{jumlah:,}', ha='center', va='bottom')
plt.ylabel('Jumlah Anggota')
plt.title('Ukuran Tiap Cluster (dari 48.842 data)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_EVAL, '4.5_ukuran_cluster.png'), bbox_inches='tight')
plt.close()

# Grafik 3: heatmap profil cluster
plt.figure(figsize=(8, 7))
sns.heatmap(tabel_profil.astype(float), annot=True, fmt='.1f', cmap='YlGnBu',
            cbar_kws={'label': 'Persen anggota sesuai kategori dominan'})
plt.title('Profil Tiap Cluster\n(persen anggota sesuai kategori dominan)')
plt.ylabel('Fitur')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_EVAL, '4.5_profil_cluster.png'), bbox_inches='tight')
plt.close()

# Grafik 4: heatmap kestabilan
plt.figure(figsize=(7, 6))
sns.heatmap(ari, annot=True, fmt='.3f', cmap='rocket_r', vmin=0, vmax=1,
            xticklabels=daftar_seed, yticklabels=daftar_seed,
            cbar_kws={'label': 'Adjusted Rand Index'})
plt.xlabel('Seed')
plt.ylabel('Seed')
plt.title('Kestabilan Cluster Antar Seed')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_EVAL, '4.5_kestabilan_ari.png'), bbox_inches='tight')
plt.close()

# 14g # Menyimpan rangkuman evaluasi ke CSV
df_evaluasi = pd.DataFrame({
    'Cluster': [f'Cluster {c}' for c in daftar_cluster],
    'Jumlah_Anggota': jumlah_anggota,
    'Persen_Anggota': [round(j / len(labels2) * 100, 2) for j in jumlah_anggota],
    'Silhouette': [round(s, 4) for s in nilai_sil],
})
df_evaluasi.to_csv(os.path.join(OUTPUT_EVAL, '4.5_rangkuman_evaluasi.csv'), index=False)

print("\n=== RINGKASAN EVALUASI ===")
print(df_evaluasi.to_string(index=False))
print(f"\nSilhouette Score (Hamming) : {silhouette_hamming:.4f}")
print(f"Cost model akhir (k=3)     : {algorithm.cost_:,.2f}")
print(f"Rata-rata ARI antar seed   : {ari[np.triu_indices(len(daftar_seed), 1)].mean():.4f}")
print(f"\nGrafik evaluasi tersimpan di: {OUTPUT_EVAL}")

