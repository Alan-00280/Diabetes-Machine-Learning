from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# PATH SETUP
input_path = (
    Path(__file__).resolve().parents[1]
    / "2-missing-and-duplication"
    / "output"
    / "cleaned-diabetes.csv"
)
output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)


# LOAD DATA
df = pd.read_csv(input_path)

print("=" * 60)
print("  TAHAP 3 — DETEKSI & PENANGANAN OUTLIER + SPLIT DATA")
print("=" * 60)
print(f"\n[INPUT] File   : {input_path}")
print(f"[INPUT] Ukuran : {df.shape[0]} baris x {df.shape[1]} kolom\n")

TARGET_COL   = "Outcome"
FEATURE_COLS = [c for c in df.columns if c != TARGET_COL]

# 1. DETEKSI OUTLIER — METODE IQR
print("=" * 60)
print("1. DETEKSI OUTLIER (Metode IQR)")
print("=" * 60)

outlier_rows = []
for col in FEATURE_COLS:
    q1          = df[col].quantile(0.25)
    q3          = df[col].quantile(0.75)
    iqr         = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    n_outliers  = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
    outlier_rows.append({
        "Fitur"      : col,
        "Q1"         : round(q1, 4),
        "Q3"         : round(q3, 4),
        "IQR"        : round(iqr, 4),
        "Lower Bound": round(lower_bound, 4),
        "Upper Bound": round(upper_bound, 4),
        "Outlier"    : n_outliers,
    })

outlier_summary = __import__("pandas").DataFrame(outlier_rows).set_index("Fitur")
print("\nRingkasan deteksi outlier per fitur:\n")
print(outlier_summary.to_string())
print(f"\nTotal outlier terdeteksi: {outlier_summary['Outlier'].sum()} data-poin\n")

# 2. VISUALISASI SEBELUM PENANGANAN
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle(
    "Boxplot Fitur — SEBELUM Penanganan Outlier", fontsize=14, fontweight="bold"
)
axes = axes.flatten()

for idx, col in enumerate(FEATURE_COLS):
    axes[idx].boxplot(
        df[col].dropna(),
        patch_artist=True,
        boxprops=dict(facecolor="#4C72B0", alpha=0.7),
    )
    axes[idx].set_title(col, fontsize=10)
    axes[idx].set_ylabel("Nilai")

for idx in range(len(FEATURE_COLS), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
ss1_path = output_dir / "boxplot-before.png"
plt.savefig(ss1_path, dpi=150)
plt.close()
print(f"[OUTPUT] Boxplot sebelum -> {ss1_path}\n")

# 3. PENANGANAN OUTLIER — CAPPING (WINSORIZATION)
#    Nilai di luar batas IQR di-clip, tidak ada baris yang dihapus.
print("=" * 60)
print("2. PENANGANAN OUTLIER (Metode Capping / Winsorization)")
print("=" * 60)

df_clean = df.copy()

for col in FEATURE_COLS:
    # Hitung batas sebelum capping
    q1_before   = df_clean[col].quantile(0.25)
    q3_before   = df_clean[col].quantile(0.75)
    iqr_before  = q3_before - q1_before
    lower_before = q1_before - 1.5 * iqr_before
    upper_before = q3_before + 1.5 * iqr_before
    before = int(((df_clean[col] < lower_before) | (df_clean[col] > upper_before)).sum())

    # Terapkan capping
    q1          = df_clean[col].quantile(0.25)
    q3          = df_clean[col].quantile(0.75)
    iqr         = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

    # Hitung batas sesudah capping
    q1_after    = df_clean[col].quantile(0.25)
    q3_after    = df_clean[col].quantile(0.75)
    iqr_after   = q3_after - q1_after
    lower_after = q1_after - 1.5 * iqr_after
    upper_after = q3_after + 1.5 * iqr_after
    after = int(((df_clean[col] < lower_after) | (df_clean[col] > upper_after)).sum())

    print(f"  {col:<20} | Outlier sebelum: {before:>3}  -> setelah: {after:>3}")

print(
    f"\nUkuran data setelah penanganan: {df_clean.shape[0]} baris "
    "(tidak ada baris yang dihapus)\n"
)

# 4. VISUALISASI SESUDAH PENANGANAN
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle(
    "Boxplot Fitur — SESUDAH Penanganan Outlier (Capping)",
    fontsize=14,
    fontweight="bold",
)
axes = axes.flatten()

for idx, col in enumerate(FEATURE_COLS):
    axes[idx].boxplot(
        df_clean[col].dropna(),
        patch_artist=True,
        boxprops=dict(facecolor="#55A868", alpha=0.7),
    )
    axes[idx].set_title(col, fontsize=10)
    axes[idx].set_ylabel("Nilai")

for idx in range(len(FEATURE_COLS), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
ss2_path = output_dir / "boxplot-after.png"
plt.savefig(ss2_path, dpi=150)
plt.close()
print(f"[OUTPUT] Boxplot sesudah -> {ss2_path}\n")

# 5. SPLIT DATA 80 / 20  (STRATIFIED)
#    stratify=Outcome menjaga proporsi kelas di kedua subset.
print("=" * 60)
print("3. SPLIT DATA 80% TRAINING / 20% TESTING")
print("=" * 60)

df_train, df_test = train_test_split(
    df_clean,
    test_size=0.20,
    random_state=42,
    stratify=df_clean[TARGET_COL],
)

df_train = df_train.reset_index(drop=True)
df_test  = df_test.reset_index(drop=True)

total   = len(df_clean)
n_train = len(df_train)
n_test  = len(df_test)

print(f"\n  Total data    : {total}")
print(f"  Training      : {n_train} ({n_train / total * 100:.1f}%)")
print(f"  Testing       : {n_test}  ({n_test / total * 100:.1f}%)")

train_dist = df_train[TARGET_COL].value_counts().sort_index()
test_dist  = df_test[TARGET_COL].value_counts().sort_index()

print("\n  Distribusi kelas (Outcome) setelah split:")
print(f"  {'Kelas':<10} {'Training':>12} {'Testing':>12}")
for cls in train_dist.index:
    print(f"  {cls:<10} {train_dist[cls]:>12} {test_dist[cls]:>12}")

# 6. VISUALISASI DISTRIBUSI SPLIT
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Distribusi Data Setelah Split 80/20", fontsize=13, fontweight="bold")

axes[0].pie(
    [n_train, n_test],
    labels=[f"Training\n({n_train})", f"Testing\n({n_test})"],
    colors=["#4C72B0", "#DD8452"],
    autopct="%1.1f%%",
    startangle=90,
)
axes[0].set_title("Proporsi Split")

axes[1].bar(
    [f"Kelas {c}" for c in train_dist.index],
    train_dist.values,
    color=["#4C72B0", "#55A868"],
)
axes[1].set_title("Distribusi Kelas — Training")
axes[1].set_ylabel("Jumlah")
for i, v in enumerate(train_dist.values):
    axes[1].text(i, v + 1, str(v), ha="center", fontsize=10)

axes[2].bar(
    [f"Kelas {c}" for c in test_dist.index],
    test_dist.values,
    color=["#4C72B0", "#55A868"],
)
axes[2].set_title("Distribusi Kelas — Testing")
axes[2].set_ylabel("Jumlah")
for i, v in enumerate(test_dist.values):
    axes[2].text(i, v + 1, str(v), ha="center", fontsize=10)

plt.tight_layout()
ss3_path = output_dir / "split-distribution.png"
plt.savefig(ss3_path, dpi=150)
plt.close()
print(f"\n[OUTPUT] Grafik distribusi split -> {ss3_path}\n")

# 7. SIMPAN OUTPUT CSV
train_path = output_dir / "train-diabetes.csv"
test_path  = output_dir / "test-diabetes.csv"

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)

print("=" * 60)
print("  RINGKASAN OUTPUT")
print("=" * 60)
print(f"  {train_path.name:<32} -> {n_train} baris")
print(f"  {test_path.name:<32} -> {n_test} baris")
print(f"  {ss1_path.name:<32} -> boxplot sebelum penanganan")
print(f"  {ss2_path.name:<32} -> boxplot sesudah penanganan")
print(f"  {ss3_path.name:<32} -> distribusi split 80/20")
print("\nTahap 3 selesai. selesai")
