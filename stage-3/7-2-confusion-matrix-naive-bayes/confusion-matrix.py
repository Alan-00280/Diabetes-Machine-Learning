"""
Stage 3 - Tahap 7: Confusion Matrix dan Metrik Evaluasi Naive Bayes
Anggota C: Raka

Input:
  - stage-2/6-2-naive-bayes-and-testing/output/test-predictions.csv
  - stage-2/6-2-naive-bayes-and-testing/output/model-metrics.csv (verifikasi)

Output (folder output/):
  - evaluation-metrics.csv       : Rincian sel CM dan 4 metrik evaluasi
  - confusion-matrix-persen.png  : Visualisasi CM dalam persen per baris
  - perbandingan-metrik.png      : Diagram batang 4 metrik antarskenario
  - perbandingan-kesalahan.png   : Diagram analisis kesalahan (Alarm Palsu vs Luput)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# 1. MENENTUKAN LOKASI FILE
folder_script = Path(__file__).resolve().parent
folder_root   = folder_script.parents[1]
input_dir     = folder_root / "stage-2" / "6-2-naive-bayes-and-testing" / "output"
output_dir    = folder_script / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# 2. MENGAMBIL DATA PREDIKSI DARI STAGE 2
prediksi = pd.read_csv(input_dir / "test-predictions.csv")

print("=" * 70)
print("  STAGE 3 - TAHAP 7: CONFUSION MATRIX & EVALUASI NAIVE BAYES")
print("  Anggota: Raka")
print("=" * 70)
print(f"Jumlah data uji : {len(prediksi)} sampel")
print("\nCuplikan Data Prediksi:")
print(prediksi.head())

KUNCI = "Aktual (y_test)"
MODEL = {
    "Original": "Pred_Original",
    "SMOTE": "Pred_SMOTE",
    "RUS": "Pred_RUS",
}

# 3. MENGHITUNG CONFUSION MATRIX DAN METRIK TIAP SKENARIO
# Definisi sel Confusion Matrix (Konteks Medis Diabetes):
#   Benar Sehat (True Negative)     = Aktual Sehat,    Prediksi Sehat
#   Alarm Palsu (False Positive)    = Aktual Sehat,    Prediksi Diabetes
#   Luput (False Negative)          = Aktual Diabetes, Prediksi Sehat
#   Benar Diabetes (True Positive)  = Aktual Diabetes, Prediksi Diabetes
hasil = []

for nama, kolom in MODEL.items():
    kunci = prediksi[KUNCI]
    ramalan = prediksi[kolom]

    cm = confusion_matrix(kunci, ramalan)
    benar_sehat = int(cm[0, 0])
    alarm_palsu = int(cm[0, 1])
    luput = int(cm[1, 0])
    benar_diabetes = int(cm[1, 1])

    accuracy  = accuracy_score(kunci, ramalan)
    precision = precision_score(kunci, ramalan, zero_division=0)
    recall    = recall_score(kunci, ramalan, zero_division=0)
    f1        = f1_score(kunci, ramalan, zero_division=0)

    hasil.append({
        "Model": nama,
        "Benar_Sehat": benar_sehat,
        "Alarm_Palsu": alarm_palsu,
        "Luput": luput,
        "Benar_Diabetes": benar_diabetes,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
    })

    print(f"\n" + "-" * 40)
    print(f"=== Skenario Naive Bayes: {nama} ===")
    print(f"  Benar Sehat (TN)    : {benar_sehat}")
    print(f"  Alarm Palsu (FP)    : {alarm_palsu}")
    print(f"  Luput (FN)          : {luput}")
    print(f"  Benar Diabetes (TP) : {benar_diabetes}")
    print("  Perhitungan Metrik:")
    print(f"    Accuracy  = ({benar_sehat} + {benar_diabetes}) / {len(kunci)} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    Precision = {benar_diabetes} / ({benar_diabetes} + {alarm_palsu}) = {precision:.4f} ({precision*100:.2f}%)")
    print(f"    Recall    = {benar_diabetes} / ({benar_diabetes} + {luput}) = {recall:.4f} ({recall*100:.2f}%)")
    print(f"    F1-Score  = 2 * (P * R) / (P + R) = {f1:.4f} ({f1*100:.2f}%)")

tabel = pd.DataFrame(hasil)
tabel.to_csv(output_dir / "evaluation-metrics.csv", index=False)

print("\n" + "=" * 70)
print("=== TABEL METRIK EVALUASI NAIVE BAYES TIAP SKENARIO ===")
print("=" * 70)
print(tabel.to_string(index=False))
print("\n[OUTPUT] evaluation-metrics.csv berhasil disimpan.")

# 4. CONFUSION MATRIX DALAM PERSEN (NORMALISASI PER BARIS AKTUAL)
# Normalisasi per baris memperlihatkan performa prediktif secara adil
# mengingat kelas aktual bersifat tidak seimbang (98 sehat vs 51 diabetes).
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Confusion Matrix Naive Bayes per Skenario (Persentase per Baris Aktual)", fontsize=13, fontweight="bold")

for ax, (nama, kolom) in zip(axes, MODEL.items()):
    cm = confusion_matrix(prediksi[KUNCI], prediksi[kolom])
    cm_persen = cm / cm.sum(axis=1, keepdims=True) * 100

    ax.imshow(cm_persen, cmap="Greens", vmin=0, vmax=100)
    ax.set_title(f"Naive Bayes ({nama})", fontweight="bold", fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Prediksi\nTidak Diabetes", "Prediksi\nDiabetes"], fontsize=9)
    ax.set_yticklabels(["Aktual\nTidak Diabetes", "Aktual\nDiabetes"], fontsize=9)

    for baris in range(2):
        for kolom_sel in range(2):
            ax.text(
                kolom_sel, baris,
                f"{cm_persen[baris, kolom_sel]:.1f}%\n({cm[baris, kolom_sel]})",
                ha="center", va="center", fontsize=10,
                color="white" if cm_persen[baris, kolom_sel] > 50 else "black",
                fontweight="bold" if cm_persen[baris, kolom_sel] > 50 else "normal",
            )

plt.tight_layout()
plt.savefig(output_dir / "confusion-matrix-persen.png", dpi=150)
plt.close()
print("[OUTPUT] confusion-matrix-persen.png tersimpan.")

# 5. GRAFIK PERBANDINGAN METRIK LENGKAP
fig, ax = plt.subplots(figsize=(10, 5))
METRIK = ["Accuracy", "Precision", "Recall", "F1-Score"]
warna_metrik = ["#2b5c8f", "#3e8e7e", "#e27c38", "#d9534f"]
lebar = 0.18

for nomor, metrik in enumerate(METRIK):
    posisi = [i + nomor * lebar for i in range(len(tabel))]
    bars = ax.bar(posisi, tabel[metrik], width=lebar, label=metrik, color=warna_metrik[nomor])
    for bar in bars:
        tinggi = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, tinggi + 0.015,
            f"{tinggi:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=0
        )

ax.set_xticks([i + 1.5 * lebar for i in range(len(tabel))])
ax.set_xticklabels([f"NB-{m}" for m in tabel["Model"]], fontsize=10, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.set_ylabel("Nilai Metrik (0.0 - 1.0)", fontsize=10)
ax.set_title("Perbandingan Kinerja Evaluasi Naive Bayes Antarskenario", fontweight="bold", fontsize=12)
ax.legend(loc="upper right", frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(output_dir / "perbandingan-metrik.png", dpi=150)
plt.close()
print("[OUTPUT] perbandingan-metrik.png tersimpan.")

# 6. GRAFIK JENIS KESALAHAN KLINIS (ALARM PALSU VS LUPUT)
# Alarm Palsu (FP) = Dampak biaya & kecemasan pasien sehat
# Luput (FN)       = Dampak bahaya fatal (pasien diabetes tidak tertangani)
fig, ax = plt.subplots(figsize=(8.5, 5))
posisi = list(range(len(tabel)))

bars_fp = ax.bar([p - 0.18 for p in posisi], tabel["Alarm_Palsu"], width=0.35,
                 color="#DD8452", label="Alarm Palsu (Sehat diprediksi Diabetes)")
bars_fn = ax.bar([p + 0.18 for p in posisi], tabel["Luput"], width=0.35,
                 color="#C44E52", label="Luput (Diabetes diprediksi Sehat / Fatal)")

for bar in bars_fp:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.5, f"{int(h)}", ha="center", va="bottom", fontweight="bold")

for bar in bars_fn:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.5, f"{int(h)}", ha="center", va="bottom", fontweight="bold")

ax.set_xticks(posisi)
ax.set_xticklabels([f"NB-{m}" for m in tabel["Model"]], fontsize=10, fontweight="bold")
ax.set_ylabel("Jumlah Pasien (dari 149 data uji)", fontsize=10)
ax.set_ylim(0, max(max(tabel["Alarm_Palsu"]), max(tabel["Luput"])) + 6)
ax.set_title("Analisis Kesalahan Prediksi Naive Bayes (Alarm Palsu vs Luput)", fontweight="bold", fontsize=12)
ax.legend(loc="upper right", frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(output_dir / "perbandingan-kesalahan.png", dpi=150)
plt.close()
print("[OUTPUT] perbandingan-kesalahan.png tersimpan.")

# 7. PENCOCOKAN DENGAN HASIL PENGUJIAN TAHAP 6
print("\n" + "=" * 70)
print("=== VERIFIKASI DENGAN model-metrics.csv (Tahap 6 Naive Bayes) ===")
print("=" * 70)
metrik_tahap6 = pd.read_csv(input_dir / "model-metrics.csv")

for _, baris in tabel.iterrows():
    resmi = metrik_tahap6.loc[metrik_tahap6["Model"] == baris["Model"]].iloc[0]
    selisih_acc = abs(baris["Accuracy"] - resmi["Accuracy"])
    selisih_f1  = abs(baris["F1-Score"] - resmi["F1-Score"])
    print(f"  [{baris['Model']:<9}] Acc Tahap 7: {baris['Accuracy']:.4f} | Tahap 6: {resmi['Accuracy']:.4f} | Diff: {selisih_acc:.4f}")
    print(f"               F1  Tahap 7: {baris['F1-Score']:.4f} | Tahap 6: {resmi['F1-Score']:.4f} | Diff: {selisih_f1:.4f}")

print("\n" + "=" * 70)
print("SELESAI! Seluruh artefak Stage 3 Naive Bayes telah dibuat:")
print(f"  1. {output_dir / 'evaluation-metrics.csv'}")
print(f"  2. {output_dir / 'confusion-matrix-persen.png'}")
print(f"  3. {output_dir / 'perbandingan-metrik.png'}")
print(f"  4. {output_dir / 'perbandingan-kesalahan.png'}")
print("=" * 70)
