"""
Stage 3 - Tahap 8: Perbandingan Kinerja Antarskenario dan Visualisasi Komprehensif

Pemilik : Luthfi Alan Perdana (NIM: 434241052)
Program : D4 Teknik Informatika, Fakultas Vokasi, Universitas Airlangga
Mata Kuliah : Praktikum Pembelajaran Mesin (UTS)

Deskripsi:
Skrip ini mengeksekusi pipeline komparasi lengkap:
1. Membaca metrik evaluasi dari Stage 3 Tahap 7 (evaluation-metrics.csv)
   dan prediksi data uji dari Stage 2 Tahap 6 (test-predictions.csv).
2. Menghasilkan visualisasi berstandar publikasi akademik (DPI 300):
   - Gambar 4.15: Confusion Matrix Skenario Baseline (gambar-4-15-cm-baseline.png)
   - Gambar 4.16: Confusion Matrix Skenario Oversampling (gambar-4-16-cm-oversampling.png)
   - Gambar 4.17: Confusion Matrix Skenario Undersampling (gambar-4-17-cm-undersampling.png)
   - Gambar 4.18: Grafik Perbandingan Kinerja Antarskenario (gambar-4-18-perbandingan-kinerja.png)
   - Visualisasi pendukung: Analisis Kesalahan Klinis (Panel B pada Gambar 4.18)
   - Visualisasi pendukung: Radar Chart Kinerja (perbandingan-radar.png)
3. Menyimpan ringkasan komparasi ke file CSV (perbandingan-skenario.csv).
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

stage3_7_dir = project_root / "stage-3" / "7-confusion-matrix" / "output"
stage2_6_dir = project_root / "stage-2" / "6-decision-tree-and-testing" / "output"
output_dir = script_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

college_dir = project_root / "college"
college_img_dir = college_dir / "images"
college_img_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  STAGE 3 - TAHAP 8: PERBANDINGAN KINERJA & VISUALISASI")
print("  Pemilik: Luthfi Alan Perdana (434241052)")
print("=" * 70)


# ==============================================================================
# 2. MEMBACA DAN MEMPROSES DATA METRIK
# ==============================================================================
metrics_file = stage3_7_dir / "evaluation-metrics.csv"
if not metrics_file.exists():
    raise FileNotFoundError(f"File metrik evaluasi tidak ditemukan: {metrics_file}")

df_metrics = pd.read_csv(metrics_file)
print(f"\n[INPUT] Membaca data evaluasi dari: {metrics_file}")
print(df_metrics.to_string(index=False))

# Mapping nama model agar selaras dengan penamaan resmi laporan
name_map = {
    "Original": "Baseline (tanpa resampling)",
    "SMOTE": "Oversampling (SMOTE)",
    "RUS": "Undersampling (RUS)",
}
df_metrics["Skenario"] = df_metrics["Model"].map(name_map)

# Menghitung metrik turunan klinis
# TN: Benar Sehat, FP: Alarm Palsu, FN: Luput, TP: Benar Diabetes
df_metrics["Total_Uji"] = (
    df_metrics["Benar_Sehat"]
    + df_metrics["Alarm_Palsu"]
    + df_metrics["Luput"]
    + df_metrics["Benar_Diabetes"]
)
df_metrics["Total_Aktual_Sehat"] = df_metrics["Benar_Sehat"] + df_metrics["Alarm_Palsu"]
df_metrics["Total_Aktual_Diabetes"] = df_metrics["Luput"] + df_metrics["Benar_Diabetes"]

# Specificity (TNR) = TN / (TN + FP)
df_metrics["Specificity"] = (
    df_metrics["Benar_Sehat"] / df_metrics["Total_Aktual_Sehat"]
).round(4)
# False Positive Rate (FPR) = FP / (TN + FP)
df_metrics["FPR"] = (
    df_metrics["Alarm_Palsu"] / df_metrics["Total_Aktual_Sehat"]
).round(4)
# False Negative Rate (FNR) = FN / (TP + FN)
df_metrics["FNR"] = (
    df_metrics["Luput"] / df_metrics["Total_Aktual_Diabetes"]
).round(4)

# Simpan tabel komparasi detail ke CSV
summary_csv_path = output_dir / "perbandingan-skenario.csv"
df_metrics.to_csv(summary_csv_path, index=False)
print(f"\n[OUTPUT] Tabel ringkasan komparasi disimpan di: {summary_csv_path}")


# ==============================================================================
# 3. FUNGSI MEMBUAT VISUALISASI CONFUSION MATRIX INDIVIDUAL (Gbr 4.15, 4.16, 4.17)
# ==============================================================================
def plot_individual_cm(
    tn, fp, fn, tp, scenario_title, fig_num_label, filename, cmap_palette="Blues"
):
    """
    Membuat visualisasi Confusion Matrix yang estetis, informatif, dan siap
    dimasukkan ke naskah laporan akademik.
    """
    cm = np.array([[tn, fp], [fn, tp]])
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100

    acc = (tn + tp) / (tn + fp + fn + tp)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    fig, ax = plt.subplots(figsize=(6.8, 5.8), dpi=300)

    # Plot heatmap
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

    # Teks anotasi dalam sel matriks
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
            # Warna teks adaptif terhadap intensitas background
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
    ax.set_xticklabels(
        ["Sehat (Tidak Diabetes)", "Diabetes"], fontsize=9.5, fontweight="normal"
    )
    ax.set_yticklabels(
        ["Sehat (Tidak Diabetes)", "Diabetes"],
        fontsize=9.5,
        fontweight="normal",
        rotation=0,
    )

    # Banner metrik evaluasi di bagian bawah figure
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

    # Simpan juga ke direktori images/ pada college untuk portabilitas naskah
    shutil.copy(save_path, college_img_dir / filename)
    print(f"  [BERHASIL] Dibuat: {save_path.name}")


print("\n--- Membangun Visualisasi Confusion Matrix Individual ---")
row_orig = df_metrics.loc[df_metrics["Model"] == "Original"].iloc[0]
plot_individual_cm(
    tn=int(row_orig["Benar_Sehat"]),
    fp=int(row_orig["Alarm_Palsu"]),
    fn=int(row_orig["Luput"]),
    tp=int(row_orig["Benar_Diabetes"]),
    scenario_title="Skenario 1: Baseline (Tanpa Resampling)",
    fig_num_label="Gambar 4.15",
    filename="gambar-4-15-cm-baseline.png",
    cmap_palette="Blues",
)

row_smote = df_metrics.loc[df_metrics["Model"] == "SMOTE"].iloc[0]
plot_individual_cm(
    tn=int(row_smote["Benar_Sehat"]),
    fp=int(row_smote["Alarm_Palsu"]),
    fn=int(row_smote["Luput"]),
    tp=int(row_smote["Benar_Diabetes"]),
    scenario_title="Skenario 2: Oversampling (SMOTE)",
    fig_num_label="Gambar 4.16",
    filename="gambar-4-16-cm-oversampling.png",
    cmap_palette="PuBuGn",
)

row_rus = df_metrics.loc[df_metrics["Model"] == "RUS"].iloc[0]
plot_individual_cm(
    tn=int(row_rus["Benar_Sehat"]),
    fp=int(row_rus["Alarm_Palsu"]),
    fn=int(row_rus["Luput"]),
    tp=int(row_rus["Benar_Diabetes"]),
    scenario_title="Skenario 3: Undersampling (RUS)",
    fig_num_label="Gambar 4.17",
    filename="gambar-4-17-cm-undersampling.png",
    cmap_palette="YlOrBr",
)


# ==============================================================================
# 4. FUNGSI MEMBUAT GAMBAR 4.18 (GRAFIK PERBANDINGAN KINERJA ANTARSKENARIO)
# ==============================================================================
def plot_comparison_master():
    """
    Membuat Gambar 4.18: Panel ganda yang menyajikan perbandingan 4 metrik evaluasi
    utama (Panel A) dan Analisis Distribusi Kesalahan Medis (Panel B).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), dpi=300)
    plt.subplots_adjust(wspace=0.28)

    scenarios = ["Baseline (Original)", "Oversampling (SMOTE)", "Undersampling (RUS)"]
    metrics_names = ["Akurasi", "Presisi", "Recall", "F1-Score"]

    # Warna representatif tiap metrik
    colors_metrics = ["#2563eb", "#0d9488", "#ea580c", "#16a34a"]

    # Data nilai matriks (3 skenario x 4 metrik)
    vals = [
        [
            row_orig["Accuracy"],
            row_orig["Precision"],
            row_orig["Recall"],
            row_orig["F1-Score"],
        ],
        [
            row_smote["Accuracy"],
            row_smote["Precision"],
            row_smote["Recall"],
            row_smote["F1-Score"],
        ],
        [
            row_rus["Accuracy"],
            row_rus["Precision"],
            row_rus["Recall"],
            row_rus["F1-Score"],
        ],
    ]

    # --- PANEL A: Grouped Bar Chart Metrik Evaluasi ---
    x = np.arange(len(scenarios))
    bar_width = 0.18
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]

    for m_idx, (m_name, m_col) in enumerate(zip(metrics_names, colors_metrics)):
        m_vals = [vals[s_idx][m_idx] for s_idx in range(len(scenarios))]
        bars = ax1.bar(
            x + offsets[m_idx],
            m_vals,
            width=bar_width,
            label=m_name,
            color=m_col,
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )

        for bar in bars:
            height = bar.get_height()
            ax1.annotate(
                f"{height * 100:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                rotation=0,
            )

    ax1.set_title(
        "(A) Perbandingan 4 Metrik Evaluasi Utama",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
        color="#1f2937",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Nilai Metrik", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax1.legend(loc="lower left", framealpha=0.9, fontsize=9)

    # Highlight box untuk skenario terbaik (RUS)
    ax1.axvspan(
        1.6,
        2.4,
        facecolor="#dcfce7",
        alpha=0.35,
        zorder=1,
        linestyle="--",
        edgecolor="#16a34a",
    )
    ax1.text(
        2.0,
        0.98,
        "Skenario Terbaik\n(Recall 80.4% | F1 68.3%)",
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color="#15803d",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#ffffff",
            edgecolor="#16a34a",
            linewidth=1,
        ),
    )

    # --- PANEL B: Analisis Kesalahan Medis (Alarm Palsu vs Luput) ---
    fp_vals = [
        row_orig["Alarm_Palsu"],
        row_smote["Alarm_Palsu"],
        row_rus["Alarm_Palsu"],
    ]
    fn_vals = [row_orig["Luput"], row_smote["Luput"], row_rus["Luput"]]

    bar_width_err = 0.32
    err_offsets = [-0.18, 0.18]

    bars_fp = ax2.bar(
        x + err_offsets[0],
        fp_vals,
        width=bar_width_err,
        label="Alarm Palsu / FP (Sehat divonis Sakit)",
        color="#f59e0b",
        edgecolor="white",
        linewidth=1,
        zorder=3,
    )
    bars_fn = ax2.bar(
        x + err_offsets[1],
        fn_vals,
        width=bar_width_err,
        label="Luput / FN (Sakit divonis Sehat) [KRITIS]",
        color="#dc2626",
        edgecolor="white",
        linewidth=1,
        zorder=3,
    )

    for bar in bars_fp:
        h = bar.get_height()
        ax2.annotate(
            f"{int(h)} psn\n({h/98*100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
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
            fontsize=8.5,
            fontweight="bold",
            color="#991b1b",
        )

    ax2.set_title(
        "(B) Analisis Kesalahan Diagnosis (FP vs FN)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
        color="#1f2937",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Jumlah Pasien Uji (Total N=149)", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0, 38)
    ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8.5)

    # Anotasi panah penurunan kasus luput (FN)
    ax2.annotate(
        "Penurunan Drastis Kasus Luput (FN)\n26 → 14 → 10 pasien (-61.5%)",
        xy=(2 + err_offsets[1], 10),
        xytext=(1.0, 32),
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=-0.2",
            color="#991b1b",
            lw=1.5,
        ),
        fontsize=9,
        fontweight="bold",
        color="#991b1b",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#fee2e2",
            edgecolor="#ef4444",
            linewidth=1,
        ),
    )

    fig.suptitle(
        "Gambar 4.18: Grafik Perbandingan Kinerja Antarskenario & Analisis Kesalahan Klinis",
        fontsize=13,
        fontweight="bold",
        y=0.98,
        color="#111827",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = output_dir / "gambar-4-18-perbandingan-kinerja.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    shutil.copy(save_path, college_img_dir / "gambar-4-18-perbandingan-kinerja.png")
    # Salin juga dengan nama standar perbandingan-metrik.png
    shutil.copy(save_path, output_dir / "perbandingan-metrik.png")
    print(f"  [BERHASIL] Dibuat: {save_path.name}")


print("\n--- Membangun Visualisasi Gambar 4.18 ---")
plot_comparison_master()


# ==============================================================================
# 5. VISUALISASI PENDUKUNG: RADAR CHART MULTI-METRIK
# ==============================================================================
def plot_radar_chart():
    """
    Membuat spider/radar chart untuk memperlihatkan trade-off profil komprehensif
    antara Akurasi, Presisi, Recall, F1-Score, dan Spesifisitas.
    """
    categories = ["Akurasi", "Presisi", "Recall", "F1-Score", "Spesifisitas"]
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # tutup kurva

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True), dpi=300)

    # Style
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, fontsize=10, fontweight="bold")
    ax.set_rlabel_position(0)
    plt.yticks(
        [0.2, 0.4, 0.6, 0.8, 1.0],
        ["20%", "40%", "60%", "80%", "100%"],
        color="grey",
        size=8,
    )
    plt.ylim(0, 1.05)

    data_plot = [
        (
            "Baseline",
            [
                row_orig["Accuracy"],
                row_orig["Precision"],
                row_orig["Recall"],
                row_orig["F1-Score"],
                row_orig["Specificity"],
            ],
            "#2563eb",
        ),
        (
            "Oversampling (SMOTE)",
            [
                row_smote["Accuracy"],
                row_smote["Precision"],
                row_smote["Recall"],
                row_smote["F1-Score"],
                row_smote["Specificity"],
            ],
            "#0d9488",
        ),
        (
            "Undersampling (RUS)",
            [
                row_rus["Accuracy"],
                row_rus["Precision"],
                row_rus["Recall"],
                row_rus["F1-Score"],
                row_rus["Specificity"],
            ],
            "#ea580c",
        ),
    ]

    for label, vals_radar, color in data_plot:
        v = vals_radar + vals_radar[:1]
        ax.plot(angles, v, linewidth=2, linestyle="solid", label=label, color=color)
        ax.fill(angles, v, color=color, alpha=0.15)

    plt.title(
        "Radar Profil Metrik Kinerja Antarskenario",
        size=12,
        fontweight="bold",
        pad=18,
    )
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
    plt.tight_layout()

    save_path = output_dir / "perbandingan-radar.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(save_path, college_img_dir / "perbandingan-radar.png")
    print(f"  [BERHASIL] Dibuat: {save_path.name}")


print("\n--- Membangun Visualisasi Pendukung Radar Chart ---")
plot_radar_chart()


# ==============================================================================
# 6. SELESAI
# ==============================================================================
print("\n" + "=" * 70)
print("  TAHAP 8 SELESAI DENGAN SUKSES!")
print("  - Seluruh grafik komparasi tersimpan di: stage-3/8-compare-and-visualitation/output/")
print("  - Salinan grafik laporan tersimpan di: college/images/")
print("  - File CSV perbandingan tersimpan di: stage-3/8-compare-and-visualitation/output/perbandingan-skenario.csv")
print("=" * 70)
