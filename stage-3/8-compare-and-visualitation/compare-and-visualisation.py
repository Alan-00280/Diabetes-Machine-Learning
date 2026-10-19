"""
Stage 3 - Tahap 8: Perbandingan Kinerja Komprehensif Antar-Model
(Decision Tree, Naive Bayes, dan Support Vector Machine)

Pemilik : Luthfi Alan Perdana (NIM: 434241052)
Program : D4 Teknik Informatika, Fakultas Vokasi, Universitas Airlangga
Mata Kuliah : Praktikum Pembelajaran Mesin (UTS)

Deskripsi:
Skrip ini mengeksekusi pipeline komparasi lengkap untuk 3 algoritma klasifikasi:
1. Decision Tree (Tahap 7-1)
2. Naive Bayes (Tahap 7-2)
3. Support Vector Machine / SVM (Tahap 7-3)

Masing-masing dievaluasi pada 3 skenario penanganan data tidak seimbang:
- Original / Baseline (Tanpa Resampling)
- SMOTE (Oversampling)
- RUS (Random Under Sampling)

Total 9 kombinasi model-skenario dibandingkan secara mendalam.

Menghasilkan visualisasi berstandar publikasi akademik (DPI 300):
- gambar-4-15-cm-baseline.png        : CM Baseline Decision Tree (kompatibilitas laporan)
- gambar-4-16-cm-oversampling.png     : CM SMOTE Decision Tree
- gambar-4-17-cm-undersampling.png    : CM RUS Decision Tree
- gambar-cm-best-3-models.png        : CM Berdampingan 3 Model Terbaik (DT vs NB vs SVM)
- gambar-4-18-perbandingan-kinerja.png: Multi-panel komparasi 3 algoritma & analisis kesalahan
- perbandingan-metrik.png            : Salinan perbandingan metrik utama
- perbandingan-model-terbaik.png     : Head-to-Head 3 Model Terbaik (RUS)
- perbandingan-kesalahan-klinis.png  : Analisis komparatif kesalahan medis (FP vs FN)
- perbandingan-radar.png             : Radar chart multi-metrik 3 algoritma terbaik
- heatmap-ringkasan-metrik.png       : Heatmap visual seluruh metrik untuk 9 konfigurasi
- perbandingan-skenario.csv / perbandingan-semua-model.csv: Rekapitulasi metrik lengkap
"""

from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# 1. SETUP PATH DAN DIREKTORI
# ==============================================================================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[1]

stage3_dt_dir = project_root / "stage-3" / "7-1-confusion-matrix-decision-tree" / "output"
stage3_nb_dir = project_root / "stage-3" / "7-2-confusion-matrix-naive-bayes" / "output"
stage3_svm_dir = project_root / "stage-3" / "7-3-confusion-matrix-svm" / "output"

output_dir = script_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

college_dir = project_root / "college"
college_img_dir = college_dir / "images"
college_img_dir.mkdir(parents=True, exist_ok=True)

print("=" * 75)
print("  STAGE 3 - TAHAP 8: PERBANDINGAN KINERJA LENGKAP ANTAR-MODEL")
print("  Decision Tree vs Naive Bayes vs Support Vector Machine (SVM)")
print("  Pemilik: Luthfi Alan Perdana (434241052)")
print("=" * 75)


# ==============================================================================
# 2. MEMBACA DAN MEMPROSES DATA METRIK EVALUASI DARI KETIGA MODEL
# ==============================================================================
def load_metrics(file_path: Path, algo_name: str) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"File metrik evaluasi tidak ditemukan: {file_path}")
    df = pd.read_csv(file_path)
    df["Algoritma"] = algo_name
    return df

df_dt = load_metrics(stage3_dt_dir / "evaluation-metrics.csv", "Decision Tree")
df_nb = load_metrics(stage3_nb_dir / "evaluation-metrics.csv", "Naive Bayes")
df_svm = load_metrics(stage3_svm_dir / "evaluation-metrics.csv", "SVM")

df_all = pd.concat([df_dt, df_nb, df_svm], ignore_index=True)

# Pemetaan nama skenario agar selaras dengan penamaan formal laporan
scenario_map = {
    "Original": "Baseline (Data Asli)",
    "SMOTE": "Oversampling (SMOTE)",
    "RUS": "Undersampling (RUS)",
}
df_all["Skenario"] = df_all["Model"].map(scenario_map)
df_all["Model_Lengkap"] = df_all["Algoritma"] + " (" + df_all["Model"] + ")"

# Menghitung metrik turunan klinis
df_all["Total_Uji"] = (
    df_all["Benar_Sehat"]
    + df_all["Alarm_Palsu"]
    + df_all["Luput"]
    + df_all["Benar_Diabetes"]
)
df_all["Total_Aktual_Sehat"] = df_all["Benar_Sehat"] + df_all["Alarm_Palsu"]
df_all["Total_Aktual_Diabetes"] = df_all["Luput"] + df_all["Benar_Diabetes"]

# Specificity (TNR) = TN / (TN + FP)
df_all["Specificity"] = (
    df_all["Benar_Sehat"] / df_all["Total_Aktual_Sehat"]
).round(4)
# False Positive Rate (FPR) = FP / (TN + FP)
df_all["FPR"] = (
    df_all["Alarm_Palsu"] / df_all["Total_Aktual_Sehat"]
).round(4)
# False Negative Rate (FNR) = FN / (TP + FN)
df_all["FNR"] = (
    df_all["Luput"] / df_all["Total_Aktual_Diabetes"]
).round(4)

# Susun urutan kolom yang rapi
col_order = [
    "Algoritma", "Model", "Skenario", "Model_Lengkap",
    "Benar_Sehat", "Alarm_Palsu", "Luput", "Benar_Diabetes",
    "Accuracy", "Precision", "Recall", "F1-Score",
    "Specificity", "FPR", "FNR", "Total_Uji"
]
df_all = df_all[col_order]

# Simpan tabel komparasi detail ke CSV
summary_csv_path = output_dir / "perbandingan-skenario.csv"
all_models_csv_path = output_dir / "perbandingan-semua-model.csv"
df_all.to_csv(summary_csv_path, index=False)
df_all.to_csv(all_models_csv_path, index=False)

print(f"\n[INPUT] Berhasil membaca metrik dari 3 algoritma (9 konfigurasi).")
print("\n" + "=" * 90)
print("=== TABEL KOMPARASI LENGKAP SELURUH MODEL (9 KONFIGURASI) ===")
print("=" * 90)
display_cols = ["Algoritma", "Model", "Accuracy", "Precision", "Recall", "F1-Score", "Alarm_Palsu", "Luput"]
print(df_all[display_cols].to_string(index=False))
print(f"\n[OUTPUT] Tabel disimpan di:\n  - {summary_csv_path}\n  - {all_models_csv_path}")


# ==============================================================================
# 3. VISUALISASI CONFUSION MATRIX INDIVIDUAL & KOMPARASI 3 MODEL TERBAIK
# ==============================================================================
def plot_individual_cm(
    tn, fp, fn, tp, scenario_title, fig_num_label, filename, cmap_palette="Blues"
):
    """Membuat visualisasi Confusion Matrix individual berstandar publikasi."""
    cm = np.array([[tn, fp], [fn, tp]])
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100

    acc = (tn + tp) / (tn + fp + fn + tp)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    fig, ax = plt.subplots(figsize=(6.8, 5.8), dpi=300)

    sns.heatmap(
        cm_percent,
        annot=False,
        cmap=cmap_palette,
        cbar=True,
        vmin=0,
        vmax=100,
        ax=ax,
        linewidths=2,
        linecolor="white",
        cbar_kws={"label": "Persentase terhadap Kelas Aktual (%)"},
    )

    cell_annotations = [
        [
            f"True Negative (TN)\n{tn} sampel\n({cm_percent[0, 0]:.2f}%)\n[Benar Sehat]",
            f"False Positive (FP)\n{fp} sampel\n({cm_percent[0, 1]:.2f}%)\n[Alarm Palsu]",
        ],
        [
            f"False Negative (FN)\n{fn} sampel\n({cm_percent[1, 0]:.2f}%)\n[Luput Terdeteksi]",
            f"True Positive (TP)\n{tp} sampel\n({cm_percent[1, 1]:.2f}%)\n[Benar Diabetes]",
        ],
    ]

    for i in range(2):
        for j in range(2):
            val_pct = cm_percent[i, j]
            text_color = "white" if val_pct > 50 else "#1a1a1a"
            ax.text(
                j + 0.5,
                i + 0.5,
                cell_annotations[i][j],
                ha="center",
                va="center",
                color=text_color,
                fontsize=9.5,
                fontweight="bold",
                linespacing=1.3,
            )

    ax.set_title(
        f"{fig_num_label}\nConfusion Matrix — {scenario_title}",
        fontsize=12,
        fontweight="bold",
        pad=14,
        color="#1f2937",
    )
    ax.set_xlabel("Label Prediksi Model", fontsize=10.5, fontweight="bold", labelpad=8)
    ax.set_ylabel("Label Aktual (Kenyataan)", fontsize=10.5, fontweight="bold", labelpad=8)
    ax.set_xticklabels(["Sehat (Tidak Diabetes)", "Diabetes"], fontsize=9.5)
    ax.set_yticklabels(["Sehat (Tidak Diabetes)", "Diabetes"], fontsize=9.5, rotation=0)

    stats_text = (
        f"Akurasi: {acc * 100:.2f}%   |   Presisi: {prec * 100:.2f}%   |   "
        f"Recall: {rec * 100:.2f}%   |   F1-Score: {f1 * 100:.2f}%"
    )
    plt.figtext(
        0.5,
        0.02,
        stats_text,
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f3f4f6",
            edgecolor="#d1d5db",
            linewidth=1.2,
        ),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    shutil.copy(save_path, college_img_dir / filename)
    print(f"  [BERHASIL] Dibuat: {save_path.name}")


print("\n--- Membangun Visualisasi Confusion Matrix Individual (Decision Tree) ---")
row_dt_orig = df_all.loc[(df_all["Algoritma"] == "Decision Tree") & (df_all["Model"] == "Original")].iloc[0]
row_dt_smote = df_all.loc[(df_all["Algoritma"] == "Decision Tree") & (df_all["Model"] == "SMOTE")].iloc[0]
row_dt_rus = df_all.loc[(df_all["Algoritma"] == "Decision Tree") & (df_all["Model"] == "RUS")].iloc[0]

plot_individual_cm(
    tn=int(row_dt_orig["Benar_Sehat"]),
    fp=int(row_dt_orig["Alarm_Palsu"]),
    fn=int(row_dt_orig["Luput"]),
    tp=int(row_dt_orig["Benar_Diabetes"]),
    scenario_title="Decision Tree (Baseline)",
    fig_num_label="Gambar 4.15",
    filename="gambar-4-15-cm-baseline.png",
    cmap_palette="Blues",
)
plot_individual_cm(
    tn=int(row_dt_smote["Benar_Sehat"]),
    fp=int(row_dt_smote["Alarm_Palsu"]),
    fn=int(row_dt_smote["Luput"]),
    tp=int(row_dt_smote["Benar_Diabetes"]),
    scenario_title="Decision Tree (SMOTE)",
    fig_num_label="Gambar 4.16",
    filename="gambar-4-16-cm-oversampling.png",
    cmap_palette="PuBuGn",
)
plot_individual_cm(
    tn=int(row_dt_rus["Benar_Sehat"]),
    fp=int(row_dt_rus["Alarm_Palsu"]),
    fn=int(row_dt_rus["Luput"]),
    tp=int(row_dt_rus["Benar_Diabetes"]),
    scenario_title="Decision Tree (RUS)",
    fig_num_label="Gambar 4.17",
    filename="gambar-4-17-cm-undersampling.png",
    cmap_palette="YlOrBr",
)


def plot_best_3_models_cm():
    """Membuat perbandingan Confusion Matrix berdampingan untuk 3 Model Terbaik (Skenario RUS)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300)
    models_info = [
        ("Decision Tree (RUS)", "Decision Tree", "Blues"),
        ("Naive Bayes (RUS)", "Naive Bayes", "Greens"),
        ("SVM (RUS)", "SVM", "Purples"),
    ]

    for ax, (title, algo, cmap_name) in zip(axes, models_info):
        row = df_all.loc[(df_all["Algoritma"] == algo) & (df_all["Model"] == "RUS")].iloc[0]
        tn, fp, fn, tp = int(row["Benar_Sehat"]), int(row["Alarm_Palsu"]), int(row["Luput"]), int(row["Benar_Diabetes"])
        cm = np.array([[tn, fp], [fn, tp]])
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(
            cm_pct, annot=False, cmap=cmap_name, cbar=False,
            vmin=0, vmax=100, ax=ax, linewidths=1.5, linecolor="white"
        )
        ax.set_title(
            f"{title}\nAcc: {row['Accuracy']*100:.1f}% | Rec: {row['Recall']*100:.1f}% | F1: {row['F1-Score']*100:.1f}%",
            fontsize=10.5, fontweight="bold", pad=10
        )
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(["Sehat", "Diabetes"], fontsize=9.5)
        ax.set_yticklabels(["Sehat", "Diabetes"], fontsize=9.5, rotation=0)
        ax.set_xlabel("Prediksi", fontsize=10, fontweight="bold")
        ax.set_ylabel("Aktual", fontsize=10, fontweight="bold")

        cells = [
            [f"TN: {tn}\n({cm_pct[0,0]:.1f}%)", f"FP: {fp}\n({cm_pct[0,1]:.1f}%)"],
            [f"FN: {fn}\n({cm_pct[1,0]:.1f}%)", f"TP: {tp}\n({cm_pct[1,1]:.1f}%)"]
        ]
        for r in range(2):
            for c in range(2):
                val = cm_pct[r, c]
                color = "white" if val > 50 else "black"
                ax.text(c + 0.5, r + 0.5, cells[r][c], ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    fig.suptitle("Perbandingan Confusion Matrix 3 Algoritma Terbaik (Skenario Resampling RUS)",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = output_dir / "gambar-cm-best-3-models.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(save_path, college_img_dir / save_path.name)
    print(f"  [BERHASIL] Dibuat: {save_path.name}")

plot_best_3_models_cm()


# ==============================================================================
# 4. FUNGSI MEMBUAT GAMBAR 4.18 (KOMPARASI LENGKAP 3 MODEL & ANALISIS KESALAHAN)
# ==============================================================================
def plot_comparison_master():
    """
    Membuat Gambar 4.18:
    Panel A: Perbandingan 4 Metrik Evaluasi (Akurasi, Presisi, Recall, F1-Score)
             antara Decision Tree, Naive Bayes, dan SVM di seluruh skenario.
    Panel B: Perbandingan Kasus Luput (FN) dan Alarm Palsu (FP) antarmodel.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    plt.subplots_adjust(wspace=0.28)

    # Label sumbu X: 3 Skenario Pembelajaran
    scenarios = ["Baseline (Original)", "Oversampling (SMOTE)", "Undersampling (RUS)"]
    algos = ["Decision Tree", "Naive Bayes", "SVM"]
    algo_colors = ["#2563eb", "#059669", "#7c3aed"]  # Blue, Green, Purple

    x = np.arange(len(scenarios))
    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]

    # --- PANEL A: F1-Score dan Recall (Dua Metrik Evaluasi Paling Vital) ---
    # Kita tampilkan F1-Score sebagai metrik agregat keseimbangan kelas
    for a_idx, (algo_name, color) in enumerate(zip(algos, algo_colors)):
        f1_vals = [
            df_all.loc[(df_all["Algoritma"] == algo_name) & (df_all["Model"] == sc_code)]["F1-Score"].values[0]
            for sc_code in ["Original", "SMOTE", "RUS"]
        ]
        rec_vals = [
            df_all.loc[(df_all["Algoritma"] == algo_name) & (df_all["Model"] == sc_code)]["Recall"].values[0]
            for sc_code in ["Original", "SMOTE", "RUS"]
        ]

        bars = ax1.bar(
            x + offsets[a_idx],
            f1_vals,
            width=bar_width,
            label=f"{algo_name}",
            color=color,
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )

        for bar, rec in zip(bars, rec_vals):
            h = bar.get_height()
            ax1.annotate(
                f"F1:{h*100:.1f}%\n(R:{rec*100:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
            )

    ax1.set_title(
        "(A) Perbandingan F1-Score & (Recall) Antarmodel per Skenario",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
        color="#1f2937",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Nilai F1-Score (0.0 - 1.0)", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(0, 1.08)
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax1.legend(loc="lower left", framealpha=0.9, fontsize=9)

    # Highlight box untuk Skenario RUS
    ax1.axvspan(
        1.55, 2.45,
        facecolor="#dcfce7", alpha=0.35, zorder=1,
        linestyle="--", edgecolor="#16a34a"
    )
    ax1.text(
        2.0, 1.02,
        "Zona Skenario RUS\n(Performa Tertinggi untuk Seluruh Algoritma)",
        ha="center", va="top", fontsize=8.2, fontweight="bold", color="#15803d",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#16a34a", linewidth=1)
    )

    # --- PANEL B: Analisis Kasus Luput / False Negative (Bahaya Fatal Klinis) ---
    for a_idx, (algo_name, color) in enumerate(zip(algos, algo_colors)):
        fn_vals = [
            df_all.loc[(df_all["Algoritma"] == algo_name) & (df_all["Model"] == sc_code)]["Luput"].values[0]
            for sc_code in ["Original", "SMOTE", "RUS"]
        ]

        bars_fn = ax2.bar(
            x + offsets[a_idx],
            fn_vals,
            width=bar_width,
            label=f"{algo_name}",
            color=color,
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )

        for bar in bars_fn:
            h = bar.get_height()
            ax2.annotate(
                f"{int(h)} psn\n({h/51*100:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
            )

    ax2.set_title(
        "(B) Kasus Pasien Diabetes Luput Terdiagnosis (False Negative / FN)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
        color="#1f2937",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Jumlah Pasien Luput (dari 51 Total Aktual Diabetes)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 36)
    ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Panah penanda reduksi luput terbaik
    ax2.annotate(
        "Minimasi Luput Terbaik:\nDT-RUS: 10 psn | SVM-RUS: 12 psn",
        xy=(2.0, 11),
        xytext=(1.05, 30),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color="#991b1b", lw=1.5),
        fontsize=8.8, fontweight="bold", color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fee2e2", edgecolor="#ef4444", linewidth=1)
    )

    fig.suptitle(
        "Gambar 4.18: Perbandingan Kinerja Antarmodel (Decision Tree, Naive Bayes, SVM) & Analisis Risiko Klinis",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
        color="#111827",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = output_dir / "gambar-4-18-perbandingan-kinerja.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    shutil.copy(save_path, college_img_dir / "gambar-4-18-perbandingan-kinerja.png")
    shutil.copy(save_path, output_dir / "perbandingan-metrik.png")
    print(f"  [BERHASIL] Dibuat: {save_path.name}")

print("\n--- Membangun Visualisasi Gambar 4.18 (Komparasi 3 Model) ---")
plot_comparison_master()

# ==============================================================================
# 5. VISUALISASI HEAD-TO-HEAD: PERBANDINGAN 3 MODEL TERBAIK (RUS)
# ==============================================================================
def plot_head_to_head_best():
    """
    Membuat grafik khusus komparasi langsung (Head-to-Head) 3 model terbaik (Skenario RUS):
    Decision Tree (RUS) vs Naive Bayes (RUS) vs SVM (RUS).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    plt.subplots_adjust(wspace=0.25)

    df_best = df_all.loc[df_all["Model"] == "RUS"].copy()
    models_label = ["Decision Tree (RUS)", "Naive Bayes (RUS)", "SVM (RUS)"]
    colors = ["#2563eb", "#059669", "#7c3aed"]

    # --- PANEL 1: 4 Metrik Evaluasi ---
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metrics_id = ["Akurasi", "Presisi", "Recall", "F1-Score"]
    x = np.arange(len(metrics))
    width = 0.24
    offsets = [-width, 0, width]

    for m_idx, (model_row, col, lbl) in enumerate(zip(df_best.iterrows(), colors, models_label)):
        vals = [model_row[1][m] for m in metrics]
        bars = ax1.bar(x + offsets[m_idx], vals, width=width, label=lbl, color=col, edgecolor="white", linewidth=1)
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h*100:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                         fontsize=8.5, fontweight="bold")

    ax1.set_title("Komparasi 4 Metrik Evaluasi Head-to-Head", fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_id, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Nilai Metrik", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(0, 0.95)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", framealpha=0.9, fontsize=9)

    # --- PANEL 2: Analisis Kesalahan FP & FN ---
    x_err = np.arange(len(models_label))
    fp_vals = df_best["Alarm_Palsu"].values
    fn_vals = df_best["Luput"].values

    bar_err_w = 0.35
    b_fp = ax2.bar(x_err - 0.18, fp_vals, width=bar_err_w, label="Alarm Palsu / FP (Sehat divonis Sakit)",
                   color="#f59e0b", edgecolor="white", linewidth=1)
    b_fn = ax2.bar(x_err + 0.18, fn_vals, width=bar_err_w, label="Luput / FN (Sakit divonis Sehat) [KRITIS]",
                   color="#dc2626", edgecolor="white", linewidth=1)

    for bar in b_fp:
        h = bar.get_height()
        ax2.annotate(f"{int(h)} psn", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

    for bar in b_fn:
        h = bar.get_height()
        ax2.annotate(f"{int(h)} psn", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color="#991b1b")

    ax2.set_title("Perbandingan Kesalahan Klinis (Alarm Palsu vs Luput)", fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_xticks(x_err)
    ax2.set_xticklabels(models_label, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Jumlah Pasien Uji (Total N=149)", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0, 36)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

    fig.suptitle("Perbandingan Head-to-Head 3 Model Terbaik: Decision Tree vs Naive Bayes vs SVM",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = output_dir / "perbandingan-model-terbaik.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(save_path, college_img_dir / save_path.name)
    print(f"  [BERHASIL] Dibuat: {save_path.name}")

print("\n--- Membangun Visualisasi Head-to-Head 3 Model Terbaik ---")
plot_head_to_head_best()


# ==============================================================================
# 6. VISUALISASI KOMPARATIF KESALAHAN KLINIS SELURUH MODEL (9 KONFIGURASI)
# ==============================================================================
def plot_clinical_errors_all():
    """Grafik perbandingan kasus luput (FN) dan alarm palsu (FP) untuk seluruh 9 model."""
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=300)

    labels = [f"{row['Algoritma']}\n({row['Model']})" for _, row in df_all.iterrows()]
    x = np.arange(len(labels))
    w = 0.38

    bars_fp = ax.bar(x - w/2, df_all["Alarm_Palsu"], width=w, label="Alarm Palsu / FP (Sehat diprediksi Sakit)",
                     color="#f59e0b", edgecolor="white", linewidth=1)
    bars_fn = ax.bar(x + w/2, df_all["Luput"], width=w, label="Luput / FN (Sakit diprediksi Sehat) [KRITIS]",
                     color="#dc2626", edgecolor="white", linewidth=1)

    for bar in bars_fp:
        h = bar.get_height()
        ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    for bar in bars_fn:
        h = bar.get_height()
        ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#991b1b")

    ax.set_title("Analisis Kesalahan Diagnostik (FP vs FN) pada 9 Konfigurasi Model", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Jumlah Pasien", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 38)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    save_path = output_dir / "perbandingan-kesalahan-klinis.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(save_path, college_img_dir / save_path.name)
    print(f"  [BERHASIL] Dibuat: {save_path.name}")

print("\n--- Membangun Visualisasi Kesalahan Klinis 9 Konfigurasi ---")
plot_clinical_errors_all()


# ==============================================================================
# 7. VISUALISASI PENDUKUNG: RADAR CHART MULTI-METRIK 3 ALGORITMA TERBAIK
# ==============================================================================
def plot_radar_chart():
    """Radar chart membandingkan profil multi-metrik dari 3 algoritma terbaik (Skenario RUS)."""
    categories = ["Akurasi", "Presisi", "Recall", "F1-Score", "Spesifisitas"]
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=300)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, fontsize=10, fontweight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=8)
    plt.ylim(0, 1.05)

    df_rus = df_all.loc[df_all["Model"] == "RUS"]
    row_dt = df_rus.loc[df_rus["Algoritma"] == "Decision Tree"].iloc[0]
    row_nb = df_rus.loc[df_rus["Algoritma"] == "Naive Bayes"].iloc[0]
    row_svm = df_rus.loc[df_rus["Algoritma"] == "SVM"].iloc[0]

    data_plot = [
        ("Decision Tree (RUS)", [row_dt["Accuracy"], row_dt["Precision"], row_dt["Recall"], row_dt["F1-Score"], row_dt["Specificity"]], "#2563eb"),
        ("Naive Bayes (RUS)", [row_nb["Accuracy"], row_nb["Precision"], row_nb["Recall"], row_nb["F1-Score"], row_nb["Specificity"]], "#059669"),
        ("SVM (RUS)", [row_svm["Accuracy"], row_svm["Precision"], row_svm["Recall"], row_svm["F1-Score"], row_svm["Specificity"]], "#7c3aed"),
    ]

    for label, vals_radar, color in data_plot:
        v = vals_radar + vals_radar[:1]
        ax.plot(angles, v, linewidth=2.2, linestyle="solid", label=label, color=color)
        ax.fill(angles, v, color=color, alpha=0.15)

    plt.title("Radar Profil Metrik Kinerja: Decision Tree vs Naive Bayes vs SVM (RUS)",
              size=11.5, fontweight="bold", pad=22)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.12), fontsize=9)
    plt.tight_layout()

    save_path = output_dir / "perbandingan-radar.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(save_path, college_img_dir / "perbandingan-radar.png")
    print(f"  [BERHASIL] Dibuat: {save_path.name}")

print("\n--- Membangun Visualisasi Radar Chart Antarmodel ---")
plot_radar_chart()


# ==============================================================================
# 8. VISUALISASI HEATMAP RINGKASAN METRIK SELURUH 9 MODEL
# ==============================================================================
def plot_heatmap_summary():
    """Heatmap matriks perbandingan metrik evaluasi untuk seluruh 9 model."""
    heatmap_df = df_all.copy()
    heatmap_df.index = heatmap_df["Model_Lengkap"]
    cols_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity"]
    cols_label = ["Akurasi", "Presisi", "Recall", "F1-Score", "Spesifisitas"]
    data = heatmap_df[cols_to_plot]

    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=300)
    sns.heatmap(
        data * 100,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Nilai Metrik (%)"},
        linewidths=1,
        linecolor="white",
        ax=ax,
        vmin=40,
        vmax=90,
    )
    # Tambahkan tanda % pada anotasi teks
    for t in ax.texts:
        t.set_text(t.get_text() + "%")
        t.set_fontweight("bold")
        t.set_fontsize(9.5)

    ax.set_title("Heatmap Ringkasan Kinerja 9 Konfigurasi Model Klasifikasi", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticklabels(cols_label, fontsize=10, fontweight="bold")
    ax.set_yticklabels(heatmap_df["Model_Lengkap"], fontsize=9.5, fontweight="bold", rotation=0)
    ax.set_xlabel("Metrik Evaluasi", fontsize=10.5, fontweight="bold", labelpad=8)
    ax.set_ylabel("Konfigurasi Model & Skenario", fontsize=10.5, fontweight="bold", labelpad=8)

    plt.tight_layout()
    save_path = output_dir / "heatmap-ringkasan-metrik.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(save_path, college_img_dir / save_path.name)
    print(f"  [BERHASIL] Dibuat: {save_path.name}")

print("\n--- Membangun Visualisasi Heatmap Ringkasan Metrik ---")
plot_heatmap_summary()


# ==============================================================================
# 9. SELESAI & RINGKASAN REKOMENDASI KLINIS
# ==============================================================================
print("\n" + "=" * 75)
print("  TAHAP 8 SELESAI DENGAN SUKSES!")
print("=" * 75)
print("  Artefak visualisasi berhasil disimpan di:")
print(f"  - Folder Output : {output_dir}")
print(f"  - Folder College: {college_img_dir}")
print("\n  Daftar Artefak:")
print("  1. gambar-4-15-cm-baseline.png        : CM Baseline Decision Tree")
print("  2. gambar-4-16-cm-oversampling.png     : CM SMOTE Decision Tree")
print("  3. gambar-4-17-cm-undersampling.png    : CM RUS Decision Tree")
print("  4. gambar-cm-best-3-models.png        : CM Berdampingan 3 Model Terbaik (RUS)")
print("  5. gambar-4-18-perbandingan-kinerja.png: Multi-Panel Komparasi 3 Model")
print("  6. perbandingan-metrik.png            : Salinan Grafik Komparasi Metrik")
print("  7. perbandingan-model-terbaik.png     : Head-to-Head Model Terbaik (RUS)")
print("  8. perbandingan-kesalahan-klinis.png  : Analisis Kesalahan Medis (FP vs FN)")
print("  9. perbandingan-radar.png             : Radar Multi-Metrik Komparasi 3 Algoritma")
print(" 10. heatmap-ringkasan-metrik.png       : Heatmap Evaluasi Seluruh Konfigurasi")
print(" 11. perbandingan-skenario.csv / perbandingan-semua-model.csv")
print("=" * 75)
