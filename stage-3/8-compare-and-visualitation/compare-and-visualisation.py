"""
Stage 3 - Tahap 8: Perbandingan Kinerja Komprehensif Antar-Algoritma dan Antar-Skenario
Pemilik : Raka (Anggota C)
Mata Kuliah: Praktikum Pembelajaran Mesin

Deskripsi:
Skrip ini mengeksekusi pipeline komparasi lengkap terhadap tiga algoritma:
1. Decision Tree (dari Stage 3 Tahap 7-1)
2. Naive Bayes (dari Stage 3 Tahap 7-2)
3. Support Vector Machine / SVM (dari Stage 3 Tahap 7-3)

Masing-masing dievaluasi pada tiga skenario penanganan dataset:
- Skenario 1 (Original / Baseline): Tanpa resampling (data asli tidak seimbang)
- Skenario 2 (Oversampling SMOTE): Penyeimbangan kelas sintetis minoritas
- Skenario 3 (Undersampling RUS): Penyeimbangan kelas pemotongan acak mayoritas

Output (folder output/):
- perbandingan-skenario-lengkap.csv  : Rekapitulasi metrik lengkap 9 konfigurasi model
- gambar-4-18-perbandingan-kinerja.png : Multi-panel perbandingan 4 metrik antarmodel
- perbandingan-metrik-semua-algoritma.png: Diagram batang terkelompok 3 algoritma
- perbandingan-kesalahan-klinis.png  : Diagram analisis klinis Alarm Palsu (FP) vs Luput (FN)
- perbandingan-radar.png             : Radar chart trade-off performa model terbaik
- tabel-ranking-model.png            : Visualisasi tabel ranking model terbaik
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. SETUP PATH DAN DIREKTORI
# ==============================================================================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[1]

stage3_dir = project_root / "stage-3"
path_dt  = stage3_dir / "7-1-confusion-matrix-decision-tree" / "output" / "evaluation-metrics.csv"
path_nb  = stage3_dir / "7-2-confusion-matrix-naive-bayes" / "output" / "evaluation-metrics.csv"
path_svm = stage3_dir / "7-3-confusion-matrix-svm" / "output" / "evaluation-metrics.csv"

output_dir = script_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 75)
print("  STAGE 3 - TAHAP 8: PERBANDINGAN KINERJA LENGKAP (DT, NB, SVM)")
print("  Pemilik: Raka (Anggota C)")
print("=" * 75)

# ==============================================================================
# 2. MEMBACA DAN MENGGABUNGKAN SELURUH METRIK EVALUASI
# ==============================================================================
for p, name in [(path_dt, "Decision Tree"), (path_nb, "Naive Bayes"), (path_svm, "SVM")]:
    if not p.exists():
        raise FileNotFoundError(f"File metrik evaluasi {name} tidak ditemukan pada: {p}")

df_dt = pd.read_csv(path_dt)
df_dt["Algoritma"] = "Decision Tree"

df_nb = pd.read_csv(path_nb)
df_nb["Algoritma"] = "Naive Bayes"

df_svm = pd.read_csv(path_svm)
df_svm["Algoritma"] = "SVM"

df_all = pd.concat([df_dt, df_nb, df_svm], ignore_index=True)

# Nama konfigurasi model gabungan
df_all["Model_Lengkap"] = df_all["Algoritma"] + " (" + df_all["Model"] + ")"

# Hitung metrik klinis turunan
df_all["Total_Uji"] = (
    df_all["Benar_Sehat"]
    + df_all["Alarm_Palsu"]
    + df_all["Luput"]
    + df_all["Benar_Diabetes"]
)
df_all["Total_Aktual_Sehat"] = df_all["Benar_Sehat"] + df_all["Alarm_Palsu"]
df_all["Total_Aktual_Diabetes"] = df_all["Luput"] + df_all["Benar_Diabetes"]

# Specificity (TNR) = TN / (TN + FP)
df_all["Specificity"] = (df_all["Benar_Sehat"] / df_all["Total_Aktual_Sehat"]).round(4)
# False Positive Rate (FPR) = FP / (TN + FP)
df_all["FPR"] = (df_all["Alarm_Palsu"] / df_all["Total_Aktual_Sehat"]).round(4)
# False Negative Rate (FNR) = FN / (TP + FN)
df_all["FNR"] = (df_all["Luput"] / df_all["Total_Aktual_Diabetes"]).round(4)

# Simpan ke CSV
csv_out = output_dir / "perbandingan-skenario-lengkap.csv"
df_all.to_csv(csv_out, index=False)
print(f"\n[OUTPUT] Tabel komparasi lengkap 9 model disimpan di: {csv_out.name}")

# Tampilkan ringkasan
print("\n" + "=" * 75)
print("TABEL RINGKASAN METRIK EVALUASI 3 ALGORITMA x 3 SKENARIO:")
print("=" * 75)
print(
    df_all[[
        "Algoritma", "Model", "Accuracy", "Precision", "Recall", "F1-Score", "Alarm_Palsu", "Luput"
    ]].to_string(index=False)
)

# ==============================================================================
# 3. VISUALISASI 1: DIAGRAM BATANG KOMPARASI 4 METRIK TIAP ALGORITMA (Gambar 4.18)
# ==============================================================================
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
fig.suptitle(
    "Gambar 4.18 Perbandingan Kinerja Algoritma (Decision Tree, Naive Bayes, SVM)\n"
    "pada Skenario Original, SMOTE, dan RUS",
    fontsize=14, fontweight="bold", y=0.98
)

metrics_list = [
    ("Accuracy", "Akurasi (Ketepatan Diagnosis Global)", axes[0, 0]),
    ("Precision", "Presisi (Ketepatan Prediksi Positif Diabetes)", axes[0, 1]),
    ("Recall", "Recall / Sensitivitas (Kemampuan Menemukan Pasien Sakit)", axes[1, 0]),
    ("F1-Score", "F1-Score (Keseimbangan Harmonis Presisi & Recall)", axes[1, 1]),
]

colors_algo = {"Decision Tree": "#2b5c8f", "Naive Bayes": "#3e8e7e", "SVM": "#d95f02"}
scenarios = ["Original", "SMOTE", "RUS"]
x_pos = np.arange(len(scenarios))
width = 0.25

for metric_col, title, ax in metrics_list:
    for i, algo in enumerate(["Decision Tree", "Naive Bayes", "SVM"]):
        sub_df = df_all[df_all["Algoritma"] == algo].set_index("Model").loc[scenarios]
        vals = sub_df[metric_col].values
        pos = x_pos + (i - 1) * width
        bars = ax.bar(pos, vals, width, label=algo, color=colors_algo[algo], alpha=0.9, edgecolor="black", linewidth=0.8)
        
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, h + 0.015,
                f"{h*100:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Nilai Metrik (0 - 1)", fontsize=9.5)
    ax.legend(loc="lower right" if metric_col != "Recall" else "lower right", fontsize=8.5, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
path_gbr_18 = output_dir / "gambar-4-18-perbandingan-kinerja.png"
plt.savefig(path_gbr_18, dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "perbandingan-metrik-semua-algoritma.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[OUTPUT] Visualisasi disimpan: {path_gbr_18.name}")

# ==============================================================================
# 4. VISUALISASI 2: ANALISIS KESALAHAN KLINIS MEDIS (Alarm Palsu vs Luput)
# ==============================================================================
fig, ax = plt.subplots(figsize=(13, 6), dpi=300)

model_labels = [f"{row['Algoritma']}\n({row['Model']})" for _, row in df_all.iterrows()]
indices = np.arange(len(df_all))
bar_w = 0.38

bars_fp = ax.bar(
    indices - bar_w/2, df_all["Alarm_Palsu"], bar_w,
    label="Alarm Palsu / FP (Pasien Sehat dituduh Diabetes)", color="#DD8452", edgecolor="black", linewidth=0.8
)
bars_fn = ax.bar(
    indices + bar_w/2, df_all["Luput"], bar_w,
    label="Luput / FN (Pasien Diabetes dinyatakan Sehat — Risiko Kritis)", color="#C44E52", edgecolor="black", linewidth=0.8
)

for b in bars_fp:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2., h + 0.6, f"{int(h)}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

for b in bars_fn:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2., h + 0.6, f"{int(h)}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#8b0000")

ax.set_title("Analisis Kesalahan Diagnostik Klinis 9 Konfigurasi Model (dari 149 Pasien Uji)", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(indices)
ax.set_xticklabels(model_labels, fontsize=8.5, rotation=0)
ax.set_ylabel("Jumlah Pasien", fontsize=10.5, fontweight="bold")
ax.set_ylim(0, 38)
ax.legend(loc="upper right", fontsize=9.5, frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Highlight model terbaik (FN terendah: DT-RUS & SVM-RUS)
ax.axvspan(1.5, 2.5, color="#2b5c8f", alpha=0.08, label="DT-RUS (FN=10)")
ax.axvspan(7.5, 8.5, color="#d95f02", alpha=0.08, label="SVM-RUS (FN=12)")

plt.tight_layout()
path_gbr_err = output_dir / "perbandingan-kesalahan-klinis.png"
plt.savefig(path_gbr_err, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[OUTPUT] Visualisasi disimpan: {path_gbr_err.name}")

# ==============================================================================
# 5. VISUALISASI 3: RADAR CHART KOMPARASI MODEL TERBAIK TIAP ALGORITMA (RUS)
# ==============================================================================
categories = ["Akurasi", "Presisi", "Recall", "F1-Score", "Spesifisitas"]
N = len(categories)

# Skenario terbaik untuk ketiga algoritma adalah RUS
best_models = {
    "Decision Tree (RUS)": df_all[(df_all["Algoritma"] == "Decision Tree") & (df_all["Model"] == "RUS")].iloc[0],
    "Naive Bayes (RUS)": df_all[(df_all["Algoritma"] == "Naive Bayes") & (df_all["Model"] == "RUS")].iloc[0],
    "SVM (RUS)": df_all[(df_all["Algoritma"] == "SVM") & (df_all["Model"] == "RUS")].iloc[0],
}

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True), dpi=300)
fig.suptitle("Radar Chart Perbandingan Multidimensi Model Terbaik (Skenario RUS)", fontsize=13, fontweight="bold", y=0.98)

colors_radar = {"Decision Tree (RUS)": "#2b5c8f", "Naive Bayes (RUS)": "#3e8e7e", "SVM (RUS)": "#d95f02"}

for name, row in best_models.items():
    values = [row["Accuracy"], row["Precision"], row["Recall"], row["F1-Score"], row["Specificity"]]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle="solid", label=name, color=colors_radar[name])
    ax.fill(angles, values, color=colors_radar[name], alpha=0.15)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories, fontsize=10, fontweight="bold")
ax.set_rlabel_position(0)
plt.yticks([0.4, 0.6, 0.8, 1.0], ["40%", "60%", "80%", "100%"], color="#666", size=8)
plt.ylim(0.35, 1.0)
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9.5)

plt.tight_layout()
path_gbr_radar = output_dir / "perbandingan-radar.png"
plt.savefig(path_gbr_radar, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[OUTPUT] Visualisasi disimpan: {path_gbr_radar.name}")

# ==============================================================================
# 6. VISUALISASI 4: TABEL RANKING MODEL
# ==============================================================================
df_rank = df_all.sort_values(by=["F1-Score", "Recall", "Accuracy"], ascending=False).reset_index(drop=True)
df_rank["Rank"] = [f"#{i+1}" for i in range(len(df_rank))]

fig, ax = plt.subplots(figsize=(11, 3.8), dpi=300)
ax.axis("off")

cols_display = ["Rank", "Algoritma", "Model", "Accuracy", "Precision", "Recall", "F1-Score", "Luput (FN)"]
table_data = []
for _, r in df_rank.iterrows():
    table_data.append([
        r["Rank"],
        r["Algoritma"],
        r["Model"],
        f"{r['Accuracy']*100:.2f}%",
        f"{r['Precision']*100:.2f}%",
        f"{r['Recall']*100:.2f}%",
        f"{r['F1-Score']*100:.2f}%",
        f"{int(r['Luput'])} pasien",
    ])

tbl = ax.table(cellText=table_data, colLabels=cols_display, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.5)

for j in range(len(cols_display)):
    tbl[0, j].set_facecolor("#1f2937")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Sorot 3 besar
tbl[1, 0].set_facecolor("#d4edda")
for j in range(len(cols_display)):
    tbl[1, j].set_facecolor("#d4edda") # Rank 1 (SVM-RUS)
    tbl[2, j].set_facecolor("#e8f4f8") # Rank 2 (DT-RUS)
    tbl[3, j].set_facecolor("#fef9e7") # Rank 3 (SVM-SMOTE)

ax.set_title("Tabel Ranking Kinerja 9 Model Klasifikasi Diabetes (Diurutkan berdasarkan F1-Score & Sensitivitas)", fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
path_gbr_tbl = output_dir / "tabel-ranking-model.png"
plt.savefig(path_gbr_tbl, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[OUTPUT] Visualisasi disimpan: {path_gbr_tbl.name}")

# ==============================================================================
# 7. RINGKASAN REKOMENDASI KLINIS
# ==============================================================================
print("\n" + "=" * 75)
print("HASIL KOMPARASI KINERJA & REKOMENDASI MODEL TERBAIK:")
print("=" * 75)
top1 = df_rank.iloc[0]
print(f"  Peringkat 1 (Overall Best): {top1['Algoritma']} ({top1['Model']})")
print(f"    - Akurasi   : {top1['Accuracy']*100:.2f}%")
print(f"    - Presisi   : {top1['Precision']*100:.2f}%")
print(f"    - Recall    : {top1['Recall']*100:.2f}%")
print(f"    - F1-Score  : {top1['F1-Score']*100:.2f}%")
print(f"    - Luput (FN): {int(top1['Luput'])} pasien")
print("\n  Peringkat 2: Decision Tree (RUS) -> Recall tertinggi 80.39% (Luput 10 pasien)")
print("  Rekomendasi Deployment: Model SVM-RUS dipilih sebagai model utama aplikasi web (Streamlit).")
print("=" * 75)
print("\nTahap 8 Selesai! Seluruh artefak telah disimpan di folder output/.")
