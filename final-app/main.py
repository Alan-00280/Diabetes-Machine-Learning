"""
Pipeline Klasifikasi Diabetes: Stage 1 -> Stage 2 -> Stage 3
Menggabungkan seluruh proses utama dari data mentah hingga evaluasi model.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# --- KONFIGURASI & PATH ---
RANDOM_STATE = 42
TARGET_COL = "Outcome"
MED_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_PATH = ROOT_DIR / "diabetes.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print(" PIPELINE END-TO-END KLASIFIKASI DIABETES (STAGE 1 - 3)")
    print("=" * 60)

    # 1. LOAD DATA & PREPROCESSING (STAGE 1)
    df = pd.read_csv(DATA_PATH)
    print(f"\n[Stage 1] Membaca data: {df.shape[0]} baris, {df.shape[1]} kolom")

    # Tangani nilai 0 pada fitur medis (missing value) dengan median
    df[MED_COLS] = df[MED_COLS].replace(0, np.nan)
    for col in MED_COLS:
        df[col] = df[col].fillna(df[col].median())

    # Hapus duplikasi
    df = df.drop_duplicates().reset_index(drop=True)

    # Tangani outlier dengan IQR Capping (Winsorization)
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    for col in feature_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

    # Split Data (80% Train, 20% Test, Stratified)
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=RANDOM_STATE, stratify=df[TARGET_COL]
    )
    train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    print(f"[Stage 1] Split data: Train = {len(train_df)}, Test = {len(test_df)}")

    # 2. TRANSFORMASI & RESAMPLING (STAGE 2)
    # Z-Score Scaling dihitung hanya dari Train untuk mencegah data leakage
    mean, std = train_df[feature_cols].mean(), train_df[feature_cols].std()
    X_train = (train_df[feature_cols] - mean) / std
    X_test = (test_df[feature_cols] - mean) / std
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    # Resampling pada data train (SMOTE dan RUS)
    X_smote, y_smote = SMOTE(
        random_state=RANDOM_STATE, k_neighbors=min(5, int(y_train.value_counts().min()) - 1)
    ).fit_resample(X_train, y_train)

    X_rus, y_rus = RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(
        X_train, y_train
    )

    scenarios = {
        "Baseline (Original)": (X_train, y_train),
        "Oversampling (SMOTE)": (X_smote, y_smote),
        "Undersampling (RUS)": (X_rus, y_rus),
    }

    # 3. TRAINING & EVALUASI MODEL (STAGE 2 & 3)
    results = []
    cms = {}

    for name, (X_tr, y_tr) in scenarios.items():
        clf = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        cms[name] = cm

        results.append({
            "Skenario": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "TN": cm[0, 0],
            "FP": cm[0, 1],
            "FN": cm[1, 0],
            "TP": cm[1, 1],
        })

    # Tampilkan & Simpan Ringkasan Evaluasi
    df_eval = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print(" HASIL EVALUASI MODEL DECISION TREE")
    print("=" * 60)
    print(df_eval[["Skenario", "Accuracy", "Precision", "Recall", "F1-Score"]].to_string(index=False))

    eval_csv_path = OUTPUT_DIR / "evaluation-summary.csv"
    df_eval.to_csv(eval_csv_path, index=False)
    print(f"\n[Output] Evaluasi disimpan: {eval_csv_path.name}")

    # 4. VISUALISASI PERBANDINGAN & CONFUSION MATRIX
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle("Evaluasi Klasifikasi Diabetes: Stage 1-3 Pipeline", fontsize=14, fontweight="bold")

    # Plot Confusion Matrix tiap skenario
    class_labels = ["Sehat", "Diabetes"]
    for i, (name, cm) in enumerate(cms.items()):
        ax = axes[i]
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        ax.set_xlabel("Prediksi")
        if i == 0:
            ax.set_ylabel("Aktual")

        for r in range(2):
            for c in range(2):
                color = "white" if cm[r, c] > cm.max() / 2 else "black"
                ax.text(c, r, str(cm[r, c]), ha="center", va="center", color=color, fontweight="bold")

    # Plot Bar Chart Metrik Komparasi
    ax_bar = axes[3]
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#2563eb", "#0d9488", "#ea580c", "#16a34a"]
    x = np.arange(len(scenarios))
    w = 0.2

    for idx, (m_col, color) in enumerate(zip(metric_cols, colors)):
        ax_bar.bar(x + (idx - 1.5) * w, df_eval[m_col], width=w, label=m_col, color=color)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["Baseline", "SMOTE", "RUS"], fontsize=9, fontweight="bold")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title("Perbandingan Metrik", fontsize=11, fontweight="bold")
    ax_bar.legend(fontsize=8, loc="lower left")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    viz_path = OUTPUT_DIR / "evaluation-visualization.png"
    plt.savefig(viz_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Output] Visualisasi disimpan: {viz_path.name}")
    print("=" * 60)
    print(" Pipeline selesai dengan sukses!")


if __name__ == "__main__":
    main()
