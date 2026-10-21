"""Helper: jalankan elbow method, print cost, simpan gambar."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output-visualisasi-dan-eksplorasi')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load & preprocess (mirror logic from main.py)
dataset_path = os.path.join(BASE_DIR, 'adult-incomes-dataset.csv')
df = pd.read_csv(dataset_path)
print("Shape:", df.shape)
print("Kolom:", df.columns.tolist())

# Handle missing
df = df.replace('?', np.nan)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Binning numerik (mirror main.py)
numeric_cols = df.select_dtypes(include=[np.number]).columns
# Capping outlier
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    df[col] = df[col].clip(lower=q1 - 1.5*iqr, upper=q3 + 1.5*iqr)

df['age'] = pd.cut(df['age'], bins=[0,25,45,65,100],
                   labels=['Muda','Dewasa','Paruh-Baya','Lansia'], right=True)
df['hours-per-week'] = pd.cut(df['hours-per-week'], bins=[0,39,40,100],
                               labels=['Part-time','Full-time','Over-time'], right=True)
df['capital-gain'] = pd.cut(df['capital-gain'], bins=[-1,0,np.inf],
                             labels=['No Gain','Gain'], right=True)
df['capital-loss'] = pd.cut(df['capital-loss'], bins=[-1,0,np.inf],
                             labels=['No Loss','Loss'], right=True)
df['educational-num'] = pd.cut(df['educational-num'], bins=[0,8,12,16],
                                labels=['Pendidikan Dasar','Pendidikan Menengah','Pendidikan Tinggi'], right=True)
df['fnlwgt'] = pd.qcut(df['fnlwgt'], q=3,
                        labels=['Low Weight','Medium Weight','High Weight'])
for col in numeric_cols:
    df[col] = df[col].astype(str)

X1 = df[['workclass','education','marital-status','occupation','relationship','race','gender']].astype(str).values
print("X1 shape:", X1.shape)

# Elbow
cost = []
for n in range(1, 6):
    km = KModes(n_clusters=n, init='Huang', n_init=5, verbose=0, random_state=111)
    km.fit(X1)
    cost.append(km.cost_)
    print(f"  k={n}: cost={km.cost_:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, 6), cost, 'o-', linewidth=2, markersize=8, color='steelblue')
plt.xlabel('Number of Clusters')
plt.ylabel('Cost')
plt.title('Elbow Method - K-Modes (Adult Income Dataset)')
plt.xticks(range(1, 6))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4.3_elbow_kmodes.png'), dpi=150)
print("Saved: 4.3_elbow_kmodes.png")

# Build final model (k=3 assumption, will adjust after seeing cost)
# Try k=3
for k_test in [2, 3, 4]:
    km_f = KModes(n_clusters=k_test, init='Huang', n_init=5, verbose=0, random_state=111)
    labels_f = km_f.fit_predict(X1)
    centroids_f = km_f.cluster_centroids_
    dist = pd.Series(labels_f).value_counts().sort_index()
    print(f"\n--- k={k_test} ---")
    print("Distribusi cluster:\n", dist.to_string())
    print("Centroids:\n", pd.DataFrame(centroids_f, columns=['workclass','education','marital-status','occupation','relationship','race','gender']))

# Final visualization with k=3
km_final = KModes(n_clusters=3, init='Huang', n_init=5, verbose=0, random_state=111)
labels2 = km_final.fit_predict(X1)
centroids2 = km_final.cluster_centroids_

X1_encoded = pd.DataFrame(X1).apply(LabelEncoder().fit_transform).values

plt.figure(figsize=(12, 7))
scatter = plt.scatter(X1_encoded[:, 0], X1_encoded[:, 1], c=labels2,
                      cmap='Set2', s=30, alpha=0.5)
plt.colorbar(scatter, label='Cluster')
plt.ylabel('Feature 2 (education - encoded)')
plt.xlabel('Feature 1 (workclass - encoded)')
plt.title('Visualisasi Cluster K-Modes (k=3) - Adult Income Dataset')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4.4_scatter_cluster_k3.png'), dpi=150)
print("Saved: 4.4_scatter_cluster_k3.png")
print("\n=== SELESAI ===")
