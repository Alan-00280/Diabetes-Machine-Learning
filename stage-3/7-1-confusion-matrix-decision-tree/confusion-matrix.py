from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

# 1. MENENTUKAN LOKASI FILE
folder_script = Path(__file__).resolve().parent
folder_root = folder_script.parents[1]
input_dir = folder_root / "stage-2" / "6-decision-tree-and-testing" / "output"
output_dir = folder_script / "output"
output_dir.mkdir(exist_ok=True)

# 2. MENGAMBIL DATA
# Berisi kunci jawaban dan hasil prediksi ketiga model pada 149 data uji
prediksi = pd.read_csv(input_dir / "test-predictions.csv")

print("=== TAHAP 7: CONFUSION MATRIX DAN METRIK EVALUASI ===")
print("Jumlah data uji :", len(prediksi))
print(prediksi.head())

KUNCI = "Aktual (y_test)"
MODEL = {
    "Original": "Pred_Original",
    "SMOTE": "Pred_SMOTE",
    "RUS": "Pred_RUS",
}

# 3. MENGHITUNG CONFUSION MATRIX DAN METRIK TIAP SKENARIO
# Aturan penamaan sel:
#   Benar Sehat    = kenyataan sehat, diprediksi sehat
#   Alarm Palsu    = kenyataan sehat, diprediksi diabetes
#   Luput          = kenyataan diabetes, diprediksi sehat
#   Benar Diabetes = kenyataan diabetes, diprediksi diabetes
hasil = []

for nama, kolom in MODEL.items():
    kunci = prediksi[KUNCI]
    ramalan = prediksi[kolom]

    cm = confusion_matrix(kunci, ramalan)
    benar_sehat = cm[0, 0]
    alarm_palsu = cm[0, 1]
    luput = cm[1, 0]
    benar_diabetes = cm[1, 1]

    accuracy = accuracy_score(kunci, ramalan)
    precision = precision_score(kunci, ramalan)
    recall = recall_score(kunci, ramalan)
    f1 = f1_score(kunci, ramalan)

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

    print(f"\n=== Skenario {nama} ===")
    print(f"  Benar Sehat    : {benar_sehat}")
    print(f"  Alarm Palsu    : {alarm_palsu}")
    print(f"  Luput          : {luput}")
    print(f"  Benar Diabetes : {benar_diabetes}")
    print("  Perhitungan metrik:")
    print(f"    Accuracy  = ({benar_sehat} + {benar_diabetes}) / {len(kunci)} = {accuracy:.4f}")
    print(f"    Precision = {benar_diabetes} / ({benar_diabetes} + {alarm_palsu}) = {precision:.4f}")
    print(f"    Recall    = {benar_diabetes} / ({benar_diabetes} + {luput}) = {recall:.4f}")
    print(f"    F1-Score  = {f1:.4f}")

tabel = pd.DataFrame(hasil)
tabel.to_csv(output_dir / "evaluation-metrics.csv", index=False)

print("\n=== TABEL METRIK TIAP SKENARIO ===")
print(tabel.to_string(index=False))
print("\n[OUTPUT] evaluation-metrics.csv tersimpan")

# 4. CONFUSION MATRIX DALAM PERSEN
# Ditampilkan dalam persen per baris karena jumlah kelas tidak seimbang
# (98 sehat dan 51 diabetes), sehingga angka mentah sulit dibandingkan.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Confusion Matrix per Skenario (persen per baris)", fontweight="bold")

for ax, (nama, kolom) in zip(axes, MODEL.items()):
    cm = confusion_matrix(prediksi[KUNCI], prediksi[kolom])
    cm_persen = cm / cm.sum(axis=1, keepdims=True) * 100

    ax.imshow(cm_persen, cmap="Blues", vmin=0, vmax=100)
    ax.set_title(nama, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Prediksi\nTidak Diabetes", "Prediksi\nDiabetes"], fontsize=8)
    ax.set_yticklabels(["Aktual\nTidak Diabetes", "Aktual\nDiabetes"], fontsize=8)

    for baris in range(2):
        for kolom_sel in range(2):
            ax.text(
                kolom_sel, baris,
                f"{cm_persen[baris, kolom_sel]:.1f}%\n({cm[baris, kolom_sel]})",
                ha="center", va="center", fontsize=10,
                color="white" if cm_persen[baris, kolom_sel] > 50 else "black",
            )

plt.tight_layout()
plt.savefig(output_dir / "confusion-matrix-persen.png", dpi=150)
plt.close()
print("[OUTPUT] confusion-matrix-persen.png tersimpan")

# 5. GRAFIK PERBANDINGAN METRIK
fig, ax = plt.subplots(figsize=(10, 5))
METRIK = ["Accuracy", "Precision", "Recall", "F1-Score"]
lebar = 0.2

for nomor, metrik in enumerate(METRIK):
    posisi = [i + nomor * lebar for i in range(len(tabel))]
    ax.bar(posisi, tabel[metrik], width=lebar, label=metrik)

ax.set_xticks([i + 1.5 * lebar for i in range(len(tabel))])
ax.set_xticklabels(tabel["Model"])
ax.set_ylim(0, 1)
ax.set_ylabel("Nilai")
ax.set_title("Perbandingan Metrik Evaluasi Tiap Skenario", fontweight="bold")
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / "perbandingan-metrik.png", dpi=150)
plt.close()
print("[OUTPUT] perbandingan-metrik.png tersimpan")

# 6. GRAFIK JENIS KESALAHAN
# Alarm palsu dan luput adalah dua jenis kesalahan yang akibatnya berbeda.
fig, ax = plt.subplots(figsize=(8, 5))
posisi = list(range(len(tabel)))

ax.bar([p - 0.18 for p in posisi], tabel["Alarm_Palsu"], width=0.35,
       color="#DD8452", label="Alarm Palsu (sehat dituduh diabetes)")
ax.bar([p + 0.18 for p in posisi], tabel["Luput"], width=0.35,
       color="#C44E52", label="Luput (diabetes dinyatakan sehat)")

ax.set_xticks(posisi)
ax.set_xticklabels(tabel["Model"])
ax.set_ylabel("Jumlah pasien")
ax.set_title("Perbandingan Jenis Kesalahan Tiap Skenario", fontweight="bold")
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / "perbandingan-kesalahan.png", dpi=150)
plt.close()
print("[OUTPUT] perbandingan-kesalahan.png tersimpan")

# 7. PENCOCOKAN DENGAN HASIL PENGUJIAN TAHAP 6
print("\n=== PENCOCOKAN DENGAN model-metrics.csv (tahap 6) ===")
metrik_tahap6 = pd.read_csv(input_dir / "model-metrics.csv")

for _, baris in tabel.iterrows():
    resmi = metrik_tahap6.loc[metrik_tahap6["Model"] == baris["Model"]].iloc[0]
    selisih = abs(baris["Accuracy"] - resmi["Accuracy"])
    print(f"  {baris['Model']:<9} accuracy tahap 7 = {baris['Accuracy']:.4f} | "
          f"tahap 6 = {resmi['Accuracy']:.4f} | selisih = {selisih:.4f}")

print("\n=== SELESAI ===")
print("Hasil disimpan di folder output:")
print("- evaluation-metrics.csv")
print("- confusion-matrix-persen.png")
print("- perbandingan-metrik.png")
print("- perbandingan-kesalahan.png")
