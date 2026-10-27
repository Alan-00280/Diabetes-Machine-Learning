# ==============================================================================
# PRAKTIKUM MACHINE LEARNING - CLUSTERING MENGGUNAKAN FUZZY C-MEANS (FCM)
# Dataset: Mall_Customers.csv
# ==============================================================================

# ==============================================================================
# 4.1 Eksplorasi Data
# ==============================================================================
# 1 # import tools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Direktori kerja dan output untuk menyimpan gambar visualisasi dan eksplorasi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output-visualisasi-dan-eksplorasi-mall-customer')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2 # import data
dataset_path = os.path.join(BASE_DIR, 'Mall_Customers.csv')
if not os.path.exists(dataset_path):
    dataset_path = os.path.join(os.path.dirname(BASE_DIR), 'Mall_Customers.csv')
if not os.path.exists(dataset_path):
    dataset_path = 'Mall_Customers.csv'
df = pd.read_csv(dataset_path)

print("5 Data Teratas:")
print(df.head())

# 3 # amati bentuk data
print("\nDimensi Data (Baris, Kolom):")
print(df.shape)

# 4 # Melihat ringkasan statistik deskriptif dari DataFrame
print("\nRingkasan Statistik Deskriptif:")
print(df.describe())

# 5 # cek null data
print("\n=== Pengecekan Missing Values ===")
print("Jumlah Missing Values:")
print(df.isnull().sum())

# 6 # cek outlier
numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
print("\n=== Pengecekan Outlier Fitur Numerik ===")
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
    print(f"- {col}: {outliers} outlier")

# Boxplot sederhana untuk melihat sebaran outlier
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[numeric_cols], palette='Set2')
plt.title('Boxplot Fitur Numerik Mall Customers')
plt.savefig(os.path.join(OUTPUT_DIR, '4.1_boxplot_outlier.png'), bbox_inches='tight')
plt.show()
plt.close()


# ==============================================================================
# 4.2 Visualisasi Data
# ==============================================================================
print("\n=== 4.2 Visualisasi Data ===")
plt.style.use('fivethirtyeight')

# 7 # amati bentuk visual masing-masing fitur
plt.figure(figsize=(15, 5))
for idx, col in enumerate(numeric_cols, 1):
    plt.subplot(1, 3, idx)
    sns.histplot(df[col], kde=True, bins=15, color='royalblue')
    plt.title(f'Distribusi {col}')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4.2_distribusi_fitur_numerik.png'), bbox_inches='tight')
plt.show()
plt.close()

# 8 # Ploting untuk mencari relasi antar fitur numerik
plt.figure(figsize=(15, 4))
pairs = [
    ('Age', 'Annual Income (k$)'),
    ('Age', 'Spending Score (1-100)'),
    ('Annual Income (k$)', 'Spending Score (1-100)')
]
for idx, (x_col, y_col) in enumerate(pairs, 1):
    plt.subplot(1, 3, idx)
    sns.scatterplot(x=x_col, y=y_col, data=df, hue='Gender', palette='Set1', s=70)
    plt.title(f'{x_col} vs {y_col}')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4.2_relasi_antar_fitur.png'), bbox_inches='tight')
plt.show()
plt.close()

# 9 # Melihat sebaran Spending Score dan Annual Income pada Gender
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Gender',
    data=df,
    s=120,
    alpha=0.8,
    palette=['blue', 'pink']
)
plt.title('Annual Income vs Spending Score berdasarkan Gender')
plt.savefig(os.path.join(OUTPUT_DIR, '4.2_income_vs_spending_by_gender.png'), bbox_inches='tight')
plt.show()
plt.close()


# ==============================================================================
# 4.3 Penentuan Parameter
# ==============================================================================
print("\n=== 4.3 Penentuan Parameter Fuzzy C-Means ===")

# 10 # Memilih fitur untuk clustering
# Fitur klasik Mall Customer segmentation: Annual Income dan Spending Score
fitur_fcm = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[fitur_fcm].values

# Standarisasi data secara sederhana agar skala kedua fitur seimbang
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# skfuzzy.cmeans memerlukan input berdimensi (n_features, n_samples)
data_fcm = X_scaled.T

# Menentukan nilai c (jumlah cluster) yang sesuai menggunakan FPC dan Elbow Method (Jm)
c_range = range(2, 11)
fpc_list = []
cost_jm_list = []

print("Mengevaluasi variasi jumlah cluster c (2 sampai 10)...")
for c in c_range:
    cntr_t, u_t, u0_t, d_t, jm_t, p_t, fpc_t = fuzz.cluster.cmeans(
        data_fcm, c=c, m=2.0, error=0.005, maxiter=300, seed=42
    )
    fpc_list.append(fpc_t)
    cost_jm_list.append(jm_t[-1])
    print(f"- c = {c}: FPC = {fpc_t:.4f}, Cost (Jm) = {jm_t[-1]:.2f}")

# Visualisasi kurva FPC dan Elbow Method
plt.figure(figsize=(14, 5))

# Plot FPC (Fuzzy Partition Coefficient)
plt.subplot(1, 2, 1)
plt.plot(list(c_range), fpc_list, 'o-', color='teal', linewidth=2)
plt.title('FPC vs Jumlah Cluster (c)')
plt.xlabel('Jumlah Cluster (c)')
plt.ylabel('FPC Score')
plt.xticks(list(c_range))

# Plot Objective Function Jm (Elbow Method)
plt.subplot(1, 2, 2)
plt.plot(list(c_range), cost_jm_list, 's-', color='crimson', linewidth=2)
plt.title('Elbow Method (Cost Jm) vs c')
plt.xlabel('Jumlah Cluster (c)')
plt.ylabel('Cost Jm')
plt.xticks(list(c_range))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4.3_penentuan_c_fpc_elbow.png'), bbox_inches='tight')
plt.show()
plt.close()


# ==============================================================================
# 4.4 Clustering (Pelatihan Fuzzy C-Means)
# ==============================================================================
print("\n=== 4.4 Pelatihan Model Fuzzy C-Means ===")

# Berdasarkan analisis Elbow dan segmentasi standar Mall Customers, dipilih c = 5
c_optimal = 5
m_fuzziness = 2.0

print(f"Melatih model FCM dengan c = {c_optimal}, m = {m_fuzziness}...")
cntr, u, u0, d, jm, p, fpc_final = fuzz.cluster.cmeans(
    data_fcm,
    c=c_optimal,
    m=m_fuzziness,
    error=0.005,
    maxiter=300,
    seed=42
)

# 11 # Ekstraksi label cluster keras dan derajat keanggotaan tertinggi
cluster_labels = np.argmax(u, axis=0)
max_membership = np.max(u, axis=0)

df['cluster'] = cluster_labels
df['max_membership'] = max_membership

# Simpan nilai derajat keanggotaan untuk masing-masing cluster
for i in range(c_optimal):
    df[f'u_cluster_{i}'] = u[i]

print(f"Model konvergen pada iterasi ke-{p}")
print(f"FPC Akhir: {fpc_final:.4f}")
print("\nJumlah Data Tiap Cluster:")
print(df['cluster'].value_counts().sort_index())


# ==============================================================================
# 4.5 Hasil Clustering
# ==============================================================================
print("\n=== 4.5 Hasil Clustering ===")

# 12 # Kembalikan titik centroid ke skala asli untuk kemudahan interpretasi
centroids_original = scaler.inverse_transform(cntr)
df_centroids = pd.DataFrame(centroids_original, columns=fitur_fcm)
df_centroids.index.name = 'Cluster'
print("\nCentroid Tiap Cluster (Skala Asli):")
print(df_centroids.round(2))

# Visualisasi Scatter Plot Hasil Clustering dengan Centroid
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['cluster'],
    cmap=plt.cm.Set1,
    s=80,
    alpha=0.7
)
# Menandai posisi Centroid dengan simbol bintang besar
plt.scatter(
    centroids_original[:, 0],
    centroids_original[:, 1],
    c='yellow',
    edgecolors='black',
    s=350,
    marker='*',
    label='Centroid'
)
plt.title(f'Hasil Clustering Fuzzy C-Means (c = {c_optimal})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, '4.5_scatter_cluster_fcm.png'), bbox_inches='tight')
plt.show()
plt.close()

# Visualisasi distribusi nilai derajat keanggotaan (Fuzzy Membership)
plt.figure(figsize=(8, 5))
sns.histplot(df['max_membership'], bins=20, kde=True, color='teal')
plt.title('Distribusi Derajat Keanggotaan Maksimum (u_max)')
plt.xlabel('Derajat Keanggotaan (u_max)')
plt.ylabel('Jumlah Data')
plt.savefig(os.path.join(OUTPUT_DIR, '4.5_distribusi_membership_fcm.png'), bbox_inches='tight')
plt.show()
plt.close()


# ==============================================================================
# 4.6 Interpretasi Cluster
# ==============================================================================
print("\n=== 4.6 Interpretasi Karakteristik Tiap Cluster ===")

# 13 # Rata-rata fitur per cluster
cluster_summary = df.groupby('cluster')[fitur_fcm].mean()
print("Rata-rata Fitur per Cluster:")
print(cluster_summary.round(2))

# Visualisasi rata-rata pendapatan dan pengeluaran per cluster
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x=cluster_summary.index, y=cluster_summary['Annual Income (k$)'], hue=cluster_summary.index, palette='Blues_d', legend=False)
plt.title('Rata-rata Annual Income per Cluster')
plt.xlabel('Cluster')

plt.subplot(1, 2, 2)
sns.barplot(x=cluster_summary.index, y=cluster_summary['Spending Score (1-100)'], hue=cluster_summary.index, palette='Reds_d', legend=False)
plt.title('Rata-rata Spending Score per Cluster')
plt.xlabel('Cluster')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4.6_interpretasi_cluster_summary.png'), bbox_inches='tight')
plt.show()
plt.close()

print("\n--- Ringkasan Persona Pelanggan Mall Berdasarkan Centroid ---")
for i in range(c_optimal):
    inc = df_centroids.loc[i, 'Annual Income (k$)']
    spn = df_centroids.loc[i, 'Spending Score (1-100)']
    
    # Menentukan kategori sifat konsumen secara intuitif
    inc_kat = "Tinggi" if inc > 70 else ("Sedang" if inc >= 40 else "Rendah")
    spn_kat = "Tinggi" if spn > 60 else ("Sedang" if spn >= 40 else "Rendah")
    
    if inc_kat == "Tinggi" and spn_kat == "Tinggi":
        label_persona = "Target Utama / Sultan (High Income, High Spending)"
    elif inc_kat == "Tinggi" and spn_kat == "Rendah":
        label_persona = "Konservatif / Hemat (High Income, Low Spending)"
    elif inc_kat == "Rendah" and spn_kat == "Tinggi":
        label_persona = "Konsumtif / Impulsif (Low Income, High Spending)"
    elif inc_kat == "Rendah" and spn_kat == "Rendah":
        label_persona = "Berhati-hati / Budget Buyer (Low Income, Low Spending)"
    else:
        label_persona = "Pelanggan Standar / Rata-rata (Middle Income, Middle Spending)"
        
    print(f"Cluster {i}: Pendapatan = {inc:.1f}k ({inc_kat}), Pengeluaran = {spn:.1f} ({spn_kat}) -> {label_persona}")


# ==============================================================================
# 4.7 Evaluasi Model
# ==============================================================================
print("\n=== 4.7 Evaluasi Model Fuzzy C-Means ===")

# 14 # Evaluasi metrik FPC dan Silhouette Score
sil_final = silhouette_score(X_scaled, cluster_labels)

print("Tabel Evaluasi Kinerja Model FCM:")
print(f"1. Fuzzy Partition Coefficient (FPC) : {fpc_final:.4f} (Ideal: mendekati 1.0)")
print(f"2. Silhouette Score                 : {sil_final:.4f} (Ideal: mendekati 1.0)")

# Simpan ringkasan evaluasi ke file teks di folder output
eval_summary_path = os.path.join(OUTPUT_DIR, 'evaluasi_model_fcm.txt')
with open(eval_summary_path, 'w') as f:
    f.write("=== RINGKASAN EVALUASI FUZZY C-MEANS (MALL CUSTOMERS) ===\n")
    f.write(f"Dataset: Mall_Customers.csv\n")
    f.write(f"Fitur: {', '.join(fitur_fcm)}\n")
    f.write(f"Jumlah Baris: {len(df)}\n")
    f.write(f"Jumlah Cluster (c): {c_optimal}\n")
    f.write(f"Fuzzifier (m): {m_fuzziness}\n")
    f.write(f"Iterasi Konvergensi: {p}\n")
    f.write(f"FPC: {fpc_final:.4f}\n")
    f.write(f"Silhouette Score: {sil_final:.4f}\n\n")
    f.write("Centroid Skala Asli:\n")
    f.write(df_centroids.to_string())
    f.write("\n\nRata-rata per Cluster:\n")
    f.write(cluster_summary.to_string())

print(f"\nRingkasan evaluasi berhasil disimpan di: {eval_summary_path}")
print("Semua proses clustering Fuzzy C-Means pada Mall Customers telah selesai!")
