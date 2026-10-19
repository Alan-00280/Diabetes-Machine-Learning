# ==============================================================================
# PRAKTIKUM MACHINE LEARNING - CLUSTERING MENGGUNAKAN K-MODES
# Dataset: adult-incomes-dataset.csv
# ==============================================================================

# 4.1 Eksplorasi Data
# 1 # import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# 2 # import data
df = pd.read_csv('adult-incomes-dataset.csv')
df.head()

# 3 # amati bentuk data
df.shape

# 4 # Melihat ringkasan statistik deskriptif dari DataFrame
#tambahkan code disini

# 5 # cek null data
#tambahkan code disini

# 6 # cek outlier
#tambahkan code disini


# 4.2 Visualisasi Data
# 7 # amati bentuk visual masing-masing fitur
plt.style.use('fivethirtyeight')
plt.figure(1 , figsize = (15 , 6))
n = 0
for x in ['workclass', 'education', 'marital-status']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.countplot(y = df[x])
    plt.title('Countplot of {}'.format(x))
 #tambahkan code disini untuk menampilkan figure

# 8 # Ploting untuk mencari relasi antara fitur kategorikal
plt.figure(1 , figsize = (15 , 10))
n = 0
for x in ['education', 'occupation']:
    n += 1
    plt.subplot(2 , 1 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.countplot(x = x, hue = 'gender', data = df)
    plt.xticks(rotation = 45)
    plt.title('Distribution of {} by Gender'.format(x))
 #tambahkan code disini untuk menampilkan figure

# 9 # Melihat sebaran fitur kategorikal pada income
plt.figure(1 , figsize = (15 , 8))
sns.countplot(x = 'workclass', hue = 'income', data = df)
plt.title('Workclass vs Income')
plt.xticks(rotation = 45)
plt.legend()
 #tambahkan code disini untuk menampilkan figure


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
