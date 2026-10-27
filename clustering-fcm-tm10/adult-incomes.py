# ==============================================================================
# PRAKTIKUM MACHINE LEARNING - CLUSTERING MENGGUNAKAN Fuzzy C-Means
# Dataset: adult-incomes-dataset.csv
# ==============================================================================

# 4.1 Eksplorasi Data
# 1 # import tools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# Direktori kerja dan output untuk menyimpan gambar visualisasi dan eksplorasi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output-visualisasi-dan-eksplorasi')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2 # import data
dataset_path = os.path.join(BASE_DIR, 'adult-incomes-dataset.csv')
if not os.path.exists(dataset_path):
    dataset_path = os.path.join(os.path.dirname(BASE_DIR), 'adult-incomes-dataset.csv')
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

# ==============================================================================
# 4.2 Visualisasi Data
# ==============================================================================
print("\n=== 4.2 Visualisasi Data ===")
plt.style.use('fivethirtyeight')

# 7 # Visualisasi distribusi masing-masing fitur numerik utama
fitur_fcm = ['age', 'educational-num', 'hours-per-week']
plt.figure(figsize=(16, 5))
for idx, col in enumerate(fitur_fcm, 1):
    plt.subplot(1, 3, idx)
    sns.histplot(df[col], kde=True, bins=25, color='royalblue')
    plt.title(f'Distribusi {col}')
    plt.xlabel(col)
    plt.ylabel('Frekuensi')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4.2_distribusi_fitur_numerik.png'), bbox_inches='tight')
plt.show()
plt.close()

# 8 # Korelasi dan relasi antar fitur numerik
plt.figure(figsize=(8, 6))
sns.heatmap(df[fitur_fcm].corr(), annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
plt.title('Matriks Korelasi Fitur Numerik FCM')
plt.savefig(os.path.join(OUTPUT_DIR, '4.2_korelasi_fitur_numerik.png'), bbox_inches='tight')
plt.show()
plt.close()

# 9 # Visualisasi sebaran fitur numerik terhadap tingkat pendapatan (Income)
plt.figure(figsize=(16, 5))
for idx, col in enumerate(fitur_fcm, 1):
    plt.subplot(1, 3, idx)
    sns.boxplot(x='income', y=col, hue='income', data=df, palette='Set2', legend=False)
    plt.title(f'{col} vs Income')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4.2_fitur_numerik_vs_income.png'), bbox_inches='tight')
plt.show()
plt.close()

# ==============================================================================
# 4.3 Penentuan Parameter
# ==============================================================================
print("\n=== 4.3 Penentuan Parameter Fuzzy C-Means ===")

# 10 # Standarisasi Fitur Numerik untuk FCM
# Algoritma FCM berbasis metrik jarak Euclidean, sehingga standarisasi skala sangat penting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[fitur_fcm])
# skfuzzy.cmeans memerlukan data input berdimensi (n_features, n_samples)
data_fcm = X_scaled.T

# 11 # Pengujian Jumlah Cluster (c) Menggunakan Fuzzy Partition Coefficient (FPC) & Objective Function (Cost / Jm)
# FPC mengukur kejelasan partisi fuzzy (rentang 0-1, semakin mendekati 1 semakin baik)
c_range = range(2, 7)
fpc_scores = []
cost_jm = []
sil_scores = []

print("Mengevaluasi variasi jumlah cluster c (2 sampai 6)...")
for c in c_range:
    cntr_temp, u_temp, u0_temp, d_temp, jm_temp, p_temp, fpc_temp = fuzz.cluster.cmeans(
        data_fcm, c=c, m=2.0, error=0.005, maxiter=300, init=None, seed=42
    )
    fpc_scores.append(fpc_temp)
    cost_jm.append(jm_temp[-1])
    
    # Estimasi Silhouette Score menggunakan hard partition (argmax u) pada subset data representatif
    hard_labels_temp = np.argmax(u_temp, axis=0)
    sil_sample = silhouette_score(X_scaled, hard_labels_temp, sample_size=10000, random_state=42)
    sil_scores.append(sil_sample)
    print(f"- c = {c}: FPC = {fpc_temp:.4f}, Cost (Jm) = {jm_temp[-1]:.2f}, Silhouette (10k) = {sil_sample:.4f}, Iterasi = {p_temp}")

# Visualisasi kurva FPC, Cost (Elbow), dan Silhouette Score
plt.figure(figsize=(18, 5))

# Plot FPC
plt.subplot(1, 3, 1)
plt.plot(list(c_range), fpc_scores, 'o-', color='teal', linewidth=2)
plt.title('Fuzzy Partition Coefficient (FPC) vs c')
plt.xlabel('Jumlah Cluster (c)')
plt.ylabel('FPC Score')
plt.xticks(list(c_range))

# Plot Objective Function Jm (Elbow Method)
plt.subplot(1, 3, 2)
plt.plot(list(c_range), cost_jm, 's-', color='crimson', linewidth=2)
plt.title('Objective Function Jm (Elbow) vs c')
plt.xlabel('Jumlah Cluster (c)')
plt.ylabel('Cost (Jm)')
plt.xticks(list(c_range))

# Plot Silhouette Score
plt.subplot(1, 3, 3)
plt.plot(list(c_range), sil_scores, 'd-', color='darkorange', linewidth=2)
plt.title('Silhouette Score vs c')
plt.xlabel('Jumlah Cluster (c)')
plt.ylabel('Silhouette Score')
plt.xticks(list(c_range))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4.3_penentuan_jumlah_cluster_fpc_elbow.png'), bbox_inches='tight')
plt.show()
plt.close()

# ==============================================================================
# 4.4 Clustering (Pelatihan Fuzzy C-Means)
# ==============================================================================
print("\n=== 4.4 Pelatihan Model Fuzzy C-Means ===")

# Berdasarkan evaluasi FPC, Silhouette Score puncak, dan titik siku Elbow Method, dipilih c = 3
c_optimal = 3
m_fuzziness = 2.0  # Fuzzifier standar literatur Bezdek (m = 2.0)
error_tol = 0.005
max_iter = 300

print(f"Menjalankan FCM dengan parameter: c = {c_optimal}, m = {m_fuzziness}, error = {error_tol}, maxiter = {max_iter}...")
cntr, u, u0, d, jm, p, fpc_final = fuzz.cluster.cmeans(
    data_fcm,
    c=c_optimal,
    m=m_fuzziness,
    error=error_tol,
    maxiter=max_iter,
    init=None,
    seed=42
)

# 12 # Ekstraksi hasil clustering
# Hard clustering labels diperoleh dari indeks nilai derajat keanggotaan maksimum (argmax)
cluster_labels = np.argmax(u, axis=0)
max_membership = np.max(u, axis=0)

df['cluster'] = cluster_labels
df['max_membership'] = max_membership

# Simpan derajat keanggotaan tiap cluster ke DataFrame
for i in range(c_optimal):
    df[f'u_cluster_{i}'] = u[i]

print(f"Konvergensi tercapai pada iterasi ke-{p}")
print(f"Fuzzy Partition Coefficient (FPC) akhir: {fpc_final:.4f}")
print("\nJumlah Data Tiap Cluster (Hard Assignment):")
print(df['cluster'].value_counts().sort_index())

# ==============================================================================
# 4.5 Hasil Clustering
# ==============================================================================
print("\n=== 4.5 Hasil Clustering ===")

# 13 # Konversi centroid dari skala standardisasi kembali ke skala asli
centroids_original = scaler.inverse_transform(cntr)
df_centroids = pd.DataFrame(centroids_original, columns=fitur_fcm)
df_centroids.index.name = 'Cluster'
print("\nCentroid Tiap Cluster (Skala Asli):")
print(df_centroids)

# 14 # Visualisasi Scatter Plot Cluster pada Fitur Asli
plt.figure(figsize=(16, 7))

# Subplot 1: age vs hours-per-week
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(
    df['age'], df['hours-per-week'],
    c=df['cluster'], cmap=plt.cm.Set1, s=25, alpha=0.4
)
# Plot Centroid
plt.scatter(
    centroids_original[:, 0], centroids_original[:, 2],
    c='yellow', edgecolors='black', s=250, marker='*', label='Centroid'
)
plt.title('FCM Clustering: Age vs Hours per Week')
plt.xlabel('Age')
plt.ylabel('Hours per Week')
plt.colorbar(scatter1, label='Cluster')
plt.legend()

# Subplot 2: educational-num vs hours-per-week
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(
    df['educational-num'], df['hours-per-week'],
    c=df['cluster'], cmap=plt.cm.Set1, s=25, alpha=0.4
)
# Plot Centroid
plt.scatter(
    centroids_original[:, 1], centroids_original[:, 2],
    c='yellow', edgecolors='black', s=250, marker='*', label='Centroid'
)
plt.title('FCM Clustering: Education Num vs Hours per Week')
plt.xlabel('Educational Num')
plt.ylabel('Hours per Week')
plt.colorbar(scatter2, label='Cluster')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4.5_scatter_cluster_fitur_asli.png'), bbox_inches='tight')
plt.show()
plt.close()

# 15 # Proyeksi 2D Menggunakan PCA untuk Visualisasi Menyeluruh
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
cntr_pca = pca.transform(cntr)

plt.figure(figsize=(10, 7))
scatter_pca = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=cluster_labels, cmap=plt.cm.Set1, s=20, alpha=0.4
)
plt.scatter(
    cntr_pca[:, 0], cntr_pca[:, 1],
    c='yellow', edgecolors='black', s=300, marker='*', label='Centroid'
)
plt.title('Proyeksi 2D PCA Hasil FCM Clustering')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(scatter_pca, label='Cluster')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, '4.5_visualisasi_pca_cluster.png'), bbox_inches='tight')
plt.show()
plt.close()

# 16 # Visualisasi Distribusi Derajat Keanggotaan (Membership Degree)
plt.figure(figsize=(12, 5))
sns.histplot(df['max_membership'], bins=30, kde=True, color='teal')
plt.title('Distribusi Derajat Keanggotaan Maksimum (Max Membership Degree)')
plt.xlabel('Derajat Keanggotaan (u_max)')
plt.ylabel('Frekuensi')
plt.axvline(0.5, color='red', linestyle='--', label='Ambang Batas Keanggotaan Dominan (0.5)')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, '4.5_distribusi_membership_fcm.png'), bbox_inches='tight')
plt.show()
plt.close()

# ==============================================================================
# 4.6 Interpretasi Cluster
# ==============================================================================
print("\n=== 4.6 Interpretasi Karakteristik Tiap Cluster ===")

# Ringkasan statistik mean per cluster untuk seluruh fitur numerik
cluster_summary = df.groupby('cluster')[fitur_fcm].mean()
print("Rata-rata Fitur per Cluster:")
print(cluster_summary)

# Proporsi tingkat pendapatan (Income <=50K vs >50K) per cluster
income_distribution = pd.crosstab(df['cluster'], df['income'], normalize='index') * 100
print("\nDistribusi Persentase Income per Cluster (%):")
print(income_distribution.round(2))

# Visualisasi interpretasi cluster terhadap Income
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
income_distribution.plot(kind='bar', stacked=True, color=['#7293CB', '#E1974C'], ax=plt.gca())
plt.title('Proporsi Pendapatan (Income) per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Persentase (%)')
plt.xticks(rotation=0)
plt.legend(title='Income')

plt.subplot(1, 2, 2)
sns.barplot(x='cluster', y='age', hue='cluster', data=df, palette='Set2', legend=False)
plt.title('Rata-rata Usia per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Rata-rata Usia')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4.6_interpretasi_cluster_income.png'), bbox_inches='tight')
plt.show()
plt.close()

print("\n--- Deskripsi Sosiologis & Profil Cluster ---")
print(f"""
1. Cluster 0: 'Young / Entry-Level Workforce'
   - Rata-rata Usia: ~{df_centroids.loc[0, 'age']:.1f} tahun (pekerja muda/awal karier).
   - Rata-rata Pendidikan: ~{df_centroids.loc[0, 'educational-num']:.1f} tahun (setara SMA/sekolah menengah).
   - Rata-rata Jam Kerja: ~{df_centroids.loc[0, 'hours-per-week']:.1f} jam/minggu (pekerja paruh waktu/standar).
   - Profil Pendapatan: Didominasi pendapatan <=50K ({income_distribution.loc[0, '<=50K']:.1f}%).

2. Cluster 1: 'Senior / Experienced Standard Workers'
   - Rata-rata Usia: ~{df_centroids.loc[1, 'age']:.1f} tahun (pekerja senior/paruh baya).
   - Rata-rata Pendidikan: ~{df_centroids.loc[1, 'educational-num']:.1f} tahun (setara SMA/sekolah menengah).
   - Rata-rata Jam Kerja: ~{df_centroids.loc[1, 'hours-per-week']:.1f} jam/minggu (pekerja purnawaktu stabil).
   - Profil Pendapatan: Memiliki proporsi pendapatan >50K sebesar {income_distribution.loc[1, '>50K']:.1f}%.

3. Cluster 2: 'Career Professionals / High Education & Long Hours'
   - Rata-rata Usia: ~{df_centroids.loc[2, 'age']:.1f} tahun (usia produktif matang).
   - Rata-rata Pendidikan: ~{df_centroids.loc[2, 'educational-num']:.1f} tahun (pendidikan tinggi / sarjana).
   - Rata-rata Jam Kerja: ~{df_centroids.loc[2, 'hours-per-week']:.1f} jam/minggu (jam kerja intensif/lembur).
   - Profil Pendapatan: Memiliki proporsi pendapatan >50K tertinggi ({income_distribution.loc[2, '>50K']:.1f}%).
""")

# ==============================================================================
# 4.7 Evaluasi Model
# ==============================================================================
print("\n=== 4.7 Evaluasi Model Fuzzy C-Means ===")

# Metrik Evaluasi:
# 1. Fuzzy Partition Coefficient (FPC): rentang [1/c, 1], semakin tinggi semakin baik
# 2. Silhouette Score: rentang [-1, 1], mengukur pemisahan antar cluster
# 3. Davies-Bouldin Index (DBI): semakin kecil semakin baik
# 4. Calinski-Harabasz Index (CHI): semakin besar semakin baik

sil_final = silhouette_score(X_scaled, cluster_labels, sample_size=10000, random_state=42)
dbi_final = davies_bouldin_score(X_scaled, cluster_labels)
chi_final = calinski_harabasz_score(X_scaled, cluster_labels)

print("Tabel Evaluasi Kinerja Model FCM:")
print(f"1. Fuzzy Partition Coefficient (FPC) : {fpc_final:.4f} (Ideal: mendekati 1.0)")
print(f"2. Silhouette Score (Sample 10k)    : {sil_final:.4f} (Ideal: mendekati 1.0)")
print(f"3. Davies-Bouldin Index (DBI)        : {dbi_final:.4f} (Ideal: semakin kecil)")
print(f"4. Calinski-Harabasz Index (CHI)     : {chi_final:.2f} (Ideal: semakin besar)")

# Simpan ringkasan hasil evaluasi ke file teks di folder output
eval_summary_path = os.path.join(OUTPUT_DIR, 'evaluasi_model_fcm.txt')
with open(eval_summary_path, 'w') as f:
    f.write("=== RINGKASAN EVALUASI CLUSTERING FUZZY C-MEANS (TM10) ===\n")
    f.write(f"Dataset: adult-incomes-dataset.csv\n")
    f.write(f"Fitur: {', '.join(fitur_fcm)}\n")
    f.write(f"Jumlah Baris: {len(df)}\n")
    f.write(f"Jumlah Cluster (c): {c_optimal}\n")
    f.write(f"Fuzzifier (m): {m_fuzziness}\n")
    f.write(f"Iterasi Konvergensi: {p}\n")
    f.write(f"FPC: {fpc_final:.4f}\n")
    f.write(f"Silhouette Score: {sil_final:.4f}\n")
    f.write(f"Davies-Bouldin Index: {dbi_final:.4f}\n")
    f.write(f"Calinski-Harabasz Index: {chi_final:.2f}\n\n")
    f.write("Centroid Skala Asli:\n")
    f.write(df_centroids.to_string())
    f.write("\n\nDistribusi Income (%):\n")
    f.write(income_distribution.to_string())

print(f"\nRingkasan evaluasi berhasil disimpan di: {eval_summary_path}")
print("Semua proses clustering Fuzzy C-Means telah selesai dijalankan!")


