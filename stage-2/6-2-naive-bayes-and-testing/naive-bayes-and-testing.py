from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.naive_bayes import GaussianNB

# 1. MENENTUKAN LOKASI FILE
folder_script = Path(__file__).resolve().parent
folder_root = folder_script.parents[1]
transform_dir = folder_root / "stage-2" / "4-data-transformation" / "output"
resample_dir = folder_root / "stage-2" / "5-resampling" / "output"
output_dir = folder_script / "output"
output_dir.mkdir(exist_ok=True)

TARGET_COL = "Outcome"

# 2. MENGAMBIL DATA
# Data uji selalu memakai test-transformed.csv milik tahap 4
df_test = pd.read_csv(transform_dir / "test-transformed.csv")

# Tiga varian data train: tanpa resampling, oversampling (SMOTE), undersampling (RUS)
datasets = {
    "Original": pd.read_csv(transform_dir / "train-transformed.csv"),
    "SMOTE": pd.read_csv(resample_dir / "train-oversampling-smote.csv"),
    "RUS": pd.read_csv(resample_dir / "train-undersampling-rus.csv"),
}

print("=== TAHAP 6-2: KLASIFIKASI NAIVE BAYES ===")
print(f"Data uji : {df_test.shape[0]} baris")
for nama, df in datasets.items():
    distribusi = dict(df[TARGET_COL].value_counts().sort_index())
    print(f"Data train {nama:<8} : {df.shape[0]} baris | kelas {distribusi}")

FEATURE_COLS = [kolom for kolom in df_test.columns if kolom != TARGET_COL]
X_test = df_test[FEATURE_COLS].values
y_test = df_test[TARGET_COL].values

# 3. MELATIH MODEL DAN MENGUJI PADA DATA UJI
# GaussianNB dipilih karena seluruh fitur bertipe angka kontinu.
# Model ini tidak memakai random seed karena prosesnya deterministik.
hasil = []
prediksi = {"Aktual (y_test)": y_test}
probabilitas = {}

for nama, df_train in datasets.items():
    X_train = df_train[FEATURE_COLS].values
    y_train = df_train[TARGET_COL].values

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    prediksi[f"Pred_{nama}"] = y_pred
    probabilitas[nama] = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    benar_sehat = int(cm[0][0])
    alarm_palsu = int(cm[0][1])
    luput = int(cm[1][0])
    benar_diabetes = int(cm[1][1])

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Sensitivitas = kemampuan menemukan pasien diabetes (sama dengan recall)
    # Spesifisitas = kemampuan mengenali pasien sehat
    sensitivitas = benar_diabetes / (benar_diabetes + luput)
    spesifisitas = benar_sehat / (benar_sehat + alarm_palsu)

    hasil.append({
        "Model": nama,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
        "Train Size": len(df_train),
    })

    print(f"\n=== Skenario {nama} ===")
    print(f"  Confusion matrix:\n{cm}")
    print(f"  Benar Sehat: {benar_sehat} | Alarm Palsu: {alarm_palsu} | "
          f"Luput: {luput} | Benar Diabetes: {benar_diabetes}")
    print("  Classification report:")
    print(classification_report(y_test, y_pred, target_names=["Tidak Diabetes", "Diabetes"]))
    print(f"  Sensitivity : {sensitivitas:.4f}")
    print(f"  Specificity : {spesifisitas:.4f}")

# 4. MENYIMPAN TABEL METRIK DAN HASIL PREDIKSI
tabel = pd.DataFrame(hasil)
tabel.to_csv(output_dir / "model-metrics.csv", index=False)

df_prediksi = pd.DataFrame(prediksi)
df_prediksi.to_csv(output_dir / "test-predictions.csv", index=False)

print("\n=== TABEL METRIK NAIVE BAYES ===")
print(tabel.to_string(index=False))
print("\n[OUTPUT] model-metrics.csv tersimpan")
print("[OUTPUT] test-predictions.csv tersimpan")

# 5. TABEL PERBANDINGAN METRIK
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.axis("off")

tabel_teks = [
    [baris["Model"], f"{baris['Accuracy']:.4f}", f"{baris['Precision']:.4f}",
     f"{baris['Recall']:.4f}", f"{baris['F1-Score']:.4f}", str(baris["Train Size"])]
    for _, baris in tabel.iterrows()
]
kolom_judul = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Train Size"]

tbl = ax.table(cellText=tabel_teks, colLabels=kolom_judul, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.8)

for kolom in range(len(kolom_judul)):
    tbl[0, kolom].set_facecolor("#2c5f8a")
    tbl[0, kolom].set_text_props(color="white", fontweight="bold")

baris_terbaik = int(tabel["F1-Score"].idxmax()) + 1
for kolom in range(len(kolom_judul)):
    tbl[baris_terbaik, kolom].set_facecolor("#d4edda")

ax.set_title("Tabel Perbandingan Metrik Naive Bayes", fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(output_dir / "ss1-metrics-table.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OUTPUT] ss1-metrics-table.png tersimpan")

# 6. CONFUSION MATRIX KETIGA SKENARIO
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Confusion Matrix Naive Bayes — 3 Varian Training Data",
             fontsize=13, fontweight="bold")

for ax, (nama, kolom) in zip(axes, {"Original": "Pred_Original",
                                    "SMOTE": "Pred_SMOTE",
                                    "RUS": "Pred_RUS"}.items()):
    cm = confusion_matrix(y_test, df_prediksi[kolom])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Tidak Diabetes", "Diabetes"],
                yticklabels=["Tidak Diabetes", "Diabetes"])
    ax.set_title(f"Training: {nama}", fontweight="bold")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")

plt.tight_layout()
plt.savefig(output_dir / "ss2-confusion-matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OUTPUT] ss2-confusion-matrix.png tersimpan")

# 7. SEBARAN PROBABILITAS PREDIKSI
# Naive Bayes menghasilkan probabilitas, sehingga dapat dilihat seberapa
# yakin model terhadap tebakannya pada tiap skenario.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Sebaran Probabilitas Prediksi Diabetes (Naive Bayes)",
             fontsize=13, fontweight="bold")

for ax, (nama, nilai) in zip(axes, probabilitas.items()):
    ax.hist(nilai[y_test == 0], bins=20, range=(0, 1), alpha=0.7,
            color="#4C72B0", label="Aktual Tidak Diabetes")
    ax.hist(nilai[y_test == 1], bins=20, range=(0, 1), alpha=0.7,
            color="#C44E52", label="Aktual Diabetes")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Training: {nama}", fontweight="bold")
    ax.set_xlabel("Probabilitas prediksi diabetes")
    ax.set_ylabel("Jumlah pasien")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / "ss3-sebaran-probabilitas.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OUTPUT] ss3-sebaran-probabilitas.png tersimpan")

print("\nTahap 6-2 selesai. Gunakan test-predictions.csv untuk tahap 7-2.")
