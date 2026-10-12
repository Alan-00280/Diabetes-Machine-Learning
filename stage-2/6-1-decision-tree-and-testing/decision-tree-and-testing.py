"""
Stage 2 - Tahap 6: Pembangunan & Pelatihan Decision Tree + Pengujian Model
Anggota C: Raka

Input:
  - stage-2/4-data-transformation/output/test-transformed.csv   (data uji dari Anggota A)
  - stage-2/5-resampling/output/train-oversampling-smote.csv    (dari Anggota B)
  - stage-2/5-resampling/output/train-undersampling-rus.csv     (dari Anggota B)
  - stage-2/4-data-transformation/output/train-transformed.csv  (baseline tanpa resampling)

Output (folder output/):
  - ss1-metrics-table.png          : tabel perbandingan metrik ketiga model
  - ss2-confusion-matrix.png       : confusion matrix ketiga model (berdampingan)
  - ss3-tree-original.png          : visualisasi pohon keputusan (baseline)
  - ss4-tree-smote.png             : visualisasi pohon keputusan (SMOTE)
  - ss5-tree-rus.png               : visualisasi pohon keputusan (RUS)
  - ss6-feature-importance.png     : perbandingan feature importance ketiga model
  - model-metrics.csv              : metrik lengkap dalam format CSV
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ─────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────
script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parents[1]

stage2_dir   = project_root / "stage-2"
transform_dir = stage2_dir / "4-data-transformation" / "output"
resample_dir  = stage2_dir / "5-resampling" / "output"

output_dir = script_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

TARGET_COL   = "Outcome"
RANDOM_STATE = 42
CLASS_NAMES  = ["Tidak Diabetes", "Diabetes"]

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
# Data uji: selalu pakai test-transformed.csv (dari Anggota A)
df_test  = pd.read_csv(transform_dir / "test-transformed.csv")

# Tiga varian data train:
# 1. Original (tanpa resampling) — baseline
# 2. SMOTE (oversampling)        — dari Anggota B
# 3. RUS (undersampling)         — dari Anggota B
datasets = {
    "Original" : pd.read_csv(transform_dir / "train-transformed.csv"),
    "SMOTE"    : pd.read_csv(resample_dir  / "train-oversampling-smote.csv"),
    "RUS"      : pd.read_csv(resample_dir  / "train-undersampling-rus.csv"),
}

X_test = df_test.drop(columns=TARGET_COL).values
y_test = df_test[TARGET_COL].values

print("=" * 65)
print("  TAHAP 6 — DECISION TREE: PELATIHAN & PENGUJIAN MODEL")
print("=" * 65)
print(f"\n[DATA UJI]  {len(df_test)} baris | {df_test.shape[1]-1} fitur")
for name, df in datasets.items():
    print(f"[TRAIN {name:<8}] {len(df)} baris | Outcome: {dict(df[TARGET_COL].value_counts().sort_index())}")

FEATURE_NAMES = [c for c in df_test.columns if c != TARGET_COL]

# ─────────────────────────────────────────────
# 1. PELATIHAN DECISION TREE (3 VARIAN)
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("1. PELATIHAN MODEL DECISION TREE")
print("=" * 65)

models    = {}
y_preds   = {}
cms       = {}
metrics   = []

for name, df_train in datasets.items():
    X_train = df_train.drop(columns=TARGET_COL).values
    y_train = df_train[TARGET_COL].values

    # Bangun Decision Tree dengan criterion entropy (ID3-style),
    # max_depth=5 untuk menghindari overfitting pada dataset kecil.
    clf = DecisionTreeClassifier(
        criterion    = "entropy",
        max_depth    = 5,
        min_samples_split = 10,
        min_samples_leaf  = 5,
        random_state = RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    models[name] = clf

    y_pred         = clf.predict(X_test)
    y_preds[name]  = y_pred
    cm             = confusion_matrix(y_test, y_pred)
    cms[name]      = cm

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    metrics.append({
        "Model"    : name,
        "Accuracy" : round(acc,  4),
        "Precision": round(prec, 4),
        "Recall"   : round(rec,  4),
        "F1-Score" : round(f1,   4),
        "Train Size": len(df_train),
        "Depth"    : clf.get_depth(),
        "Leaves"   : clf.get_n_leaves(),
    })

    print(f"\n[{name}]  kedalaman={clf.get_depth()}  daun={clf.get_n_leaves()}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv(output_dir / "model-metrics.csv", index=False)
print(f"\n[OUTPUT] model-metrics.csv tersimpan")

# ─────────────────────────────────────────────
# PREDIKSI TERHADAP DATA UJI (Gambar 4.14 Laporan)
# ─────────────────────────────────────────────
df_predictions = pd.DataFrame({
    "Aktual (y_test)": y_test,
    "Pred_Original"  : y_preds["Original"],
    "Pred_SMOTE"     : y_preds["SMOTE"],
    "Pred_RUS"       : y_preds["RUS"],
})
df_predictions.to_csv(output_dir / "test-predictions.csv", index=False)

print("\n" + "=" * 65)
print("HASIL PREDIKSI TERHADAP DATA UJI (15 Sampel Pertama)")
print("=" * 65)
print(df_predictions.head(15).to_string(index=True))
print(f"[OUTPUT] test-predictions.csv tersimpan")

# Visualisasi Tabel Cuplikan Prediksi untuk Gambar 4.14
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.axis("off")
sample_pred = df_predictions.head(10).reset_index()
pred_table_data = [
    [
        f"Sampel #{row['index'] + 1}",
        f"{row['Aktual (y_test)']} ({CLASS_NAMES[row['Aktual (y_test)']]})",
        f"{row['Pred_Original']} ({CLASS_NAMES[row['Pred_Original']]})",
        f"{row['Pred_SMOTE']} ({CLASS_NAMES[row['Pred_SMOTE']]})",
        f"{row['Pred_RUS']} ({CLASS_NAMES[row['Pred_RUS']]})",
    ]
    for _, row in sample_pred.iterrows()
]
pred_cols = ["No. Sampel", "Aktual (y_test)", "Prediksi Original", "Prediksi SMOTE", "Prediksi RUS"]
tbl_pred = ax.table(
    cellText  = pred_table_data,
    colLabels = pred_cols,
    cellLoc   = "center",
    loc       = "center",
)
tbl_pred.auto_set_font_size(False)
tbl_pred.set_fontsize(9)
tbl_pred.scale(1, 1.5)
for j in range(len(pred_cols)):
    tbl_pred[0, j].set_facecolor("#2c5f8a")
    tbl_pred[0, j].set_text_props(color="white", fontweight="bold")

ax.set_title("Cuplikan Hasil Prediksi Model Decision Tree pada Data Uji", fontsize=12, fontweight="bold", pad=10)
plt.tight_layout()
pred_img_path = output_dir / "ss-prediksi-data-uji.png"
plt.savefig(pred_img_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OUTPUT] ss-prediksi-data-uji.png tersimpan (khusus Gambar 4.14)")

# ─────────────────────────────────────────────
# 2. TABEL PERBANDINGAN METRIK (ss1)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.axis("off")

table_data = [
    [
        row["Model"],
        f"{row['Accuracy']:.4f}",
        f"{row['Precision']:.4f}",
        f"{row['Recall']:.4f}",
        f"{row['F1-Score']:.4f}",
        str(row["Train Size"]),
    ]
    for row in metrics
]
col_labels = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Train Size"]

tbl = ax.table(
    cellText   = table_data,
    colLabels  = col_labels,
    cellLoc    = "center",
    loc        = "center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.8)

# Warna header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2c5f8a")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Highlight baris dengan F1 terbaik
best_f1_idx = max(range(len(metrics)), key=lambda i: metrics[i]["F1-Score"])
for j in range(len(col_labels)):
    tbl[best_f1_idx + 1, j].set_facecolor("#d4edda")

ax.set_title(
    "Tabel Perbandingan Metrik Evaluasi Decision Tree",
    fontsize=12, fontweight="bold", pad=12,
)
plt.tight_layout()
ss1_path = output_dir / "ss1-metrics-table.png"
plt.savefig(ss1_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OUTPUT] ss1-metrics-table.png tersimpan")

# ─────────────────────────────────────────────
# 3. CONFUSION MATRIX KETIGA MODEL (ss2)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Confusion Matrix Decision Tree — 3 Varian Training Data",
             fontsize=13, fontweight="bold")

for ax, (name, cm) in zip(axes, cms.items()):
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"Training: {name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Prediksi", fontsize=10)
    ax.set_ylabel("Aktual", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black",
            )

plt.tight_layout()
ss2_path = output_dir / "ss2-confusion-matrix.png"
plt.savefig(ss2_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OUTPUT] ss2-confusion-matrix.png tersimpan")

# ─────────────────────────────────────────────
# 4. VISUALISASI POHON KEPUTUSAN (ss3, ss4, ss5)
# ─────────────────────────────────────────────
tree_filenames = {
    "Original": "ss3-tree-original.png",
    "SMOTE"   : "ss4-tree-smote.png",
    "RUS"     : "ss5-tree-rus.png",
}

for name, filename in tree_filenames.items():
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        models[name],
        feature_names = FEATURE_NAMES,
        class_names   = CLASS_NAMES,
        filled        = True,
        rounded       = True,
        fontsize      = 8,
        ax            = ax,
    )
    ax.set_title(
        f"Decision Tree — Training: {name}  "
        f"(depth={models[name].get_depth()}, leaves={models[name].get_n_leaves()})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    tree_path = output_dir / filename
    plt.savefig(tree_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OUTPUT] {filename} tersimpan")

# Khusus Laporan: Pohon Original dengan max_depth=2 agar tidak terlalu besar & terbaca jelas di laporan (Gambar 4.13)
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(
    models["Original"],
    max_depth     = 2,
    feature_names = FEATURE_NAMES,
    class_names   = CLASS_NAMES,
    filled        = True,
    rounded       = True,
    fontsize      = 10,
    ax            = ax,
)
ax.set_title(
    "Visualisasi Struktur Pohon Keputusan (Original — max_depth=2 untuk Laporan)",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
report_tree_path = output_dir / "ss3-tree-original-laporan.png"
plt.savefig(report_tree_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OUTPUT] ss3-tree-original-laporan.png tersimpan (khusus Gambar 4.13 Laporan)")

# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE PERBANDINGAN (ss6)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Feature Importance — Decision Tree per Varian Training Data",
             fontsize=13, fontweight="bold")

colors_map = {"Original": "#4C72B0", "SMOTE": "#55A868", "RUS": "#DD8452"}

for ax, (name, clf) in zip(axes, models.items()):
    importances = clf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_feat = [FEATURE_NAMES[i] for i in indices]
    sorted_imp  = importances[indices]

    bars = ax.barh(sorted_feat[::-1], sorted_imp[::-1], color=colors_map[name])
    ax.set_title(f"Training: {name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, sorted_imp[::-1]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

plt.tight_layout()
ss6_path = output_dir / "ss6-feature-importance.png"
plt.savefig(ss6_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OUTPUT] ss6-feature-importance.png tersimpan")

# ─────────────────────────────────────────────
# 6. RINGKASAN AKHIR
# ─────────────────────────────────────────────
best = df_metrics.loc[df_metrics["F1-Score"].idxmax()]

print("\n" + "=" * 65)
print("  RINGKASAN OUTPUT")
print("=" * 65)
print(df_metrics[["Model","Accuracy","Precision","Recall","F1-Score"]].to_string(index=False))
print(f"\n  Model terbaik (F1 tertinggi): {best['Model']}")
print(f"  F1-Score: {best['F1-Score']:.4f} | Accuracy: {best['Accuracy']:.4f}")
print(f"\n  Output tersimpan di: {output_dir}")
print("\nTahap 6 selesai.")

