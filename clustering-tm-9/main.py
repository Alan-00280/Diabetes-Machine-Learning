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
 #tambahkan code disini untuk menampilkan figure


# 4.4 Hasil Clustering
# 11 # Membangun K-Modes
algorithm = (KModes(n_clusters =  # isi nilai n sesuai dengan hasil elbow, init = 'Huang', n_init = 5, verbose = 0, random_state = 111) )
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
 #tambahkan code disini untuk menampilkan figure


# 4.5 Evaluasi Model
# 14 # Melihat nilai Silhouette Score
score2 = silhouette_score(X1_encoded, labels2)
print("Silhouette Score: ", score2)

