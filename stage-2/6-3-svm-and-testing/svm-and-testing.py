"""
Stage 2 - Tahap 6.3: Pembangunan & Pengujian Model Support Vector Machine (SVM)
Referensi: Pertemuan 8 - Klasifikasi.pdf

Instruksi:
File ini berisi kerangka dan arahan (panduan komentar) terstruktur 
yang mengadopsi format dan gaya modul kuliah "Pertemuan 8 - Klasifikasi".
Silakan isi bagian '# TODO' di bawah ini untuk mengimplementasikan kode Anda.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import modul machine learning dan metrik evaluasi dari scikit-learn
# (Sesuai modul Hal. 1 & 9: SVC, train_test_split, confusion_matrix, accuracy_score, dsb.)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.calibration import CalibratedClassifierCV

# ─────────────────────────────────────────────────────────────────────────────
# 1. SETUP PATH & FOLDER OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
# Menentukan path direktori secara relatif agar dinamis dan portable
script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parents[1]

# Path dataset (bisa memilih data transformasi stage-2 atau data mentah diabetes.csv)
transform_dir = project_root / "stage-2" / "4-data-transformation" / "output"
resample_dir  = project_root / "stage-2" / "5-resampling" / "output"
raw_data_path = project_root / "diabetes.csv"

# Folder output untuk menyimpan gambar/screenshot dan file metrik CSV
output_dir = script_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

TARGET_COL   = "Outcome"
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# 2. MEMBACA DATA (LOAD DATA)
# Sesuai Modul Hal. 1: dataframe = pd.read_excel(...) / pd.read_csv(...)
# ─────────────────────────────────────────────────────────────────────────────
# TODO: Muat dataset.
# Skenario Pipeline (Sama seperti Decision Tree di 6-1):
#   - Data uji selalu memakai: test-transformed.csv
#   - Tiga varian data train:
#       1) Original / Baseline : train-transformed.csv
#       2) SMOTE               : train-oversampling-smote.csv
#       3) RUS (Undersampling) : train-undersampling-rus.csv

# Contoh pemuatan data (aktifkan jika memakai pipeline 3 varian):
df_test = pd.read_csv(transform_dir / "test-transformed.csv")
datasets = {
    "Original": pd.read_csv(transform_dir / "train-transformed.csv"),
    "SMOTE":    pd.read_csv(resample_dir  / "train-oversampling-smote.csv"),
    "RUS":      pd.read_csv(resample_dir  / "train-undersampling-rus.csv"),
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. PEMODELAN DENGAN SVM (SUPPORT VECTOR MACHINE)
# Sesuai Modul Hal. 9 & 12: model = SVC(), model.fit(), y_pred = model.predict()
# ─────────────────────────────────────────────────────────────────────────────
print("PEMODELAN DENGAN SVM".center(75, "="))
# TODO: Latih model SVM (Support Vector Classifier).
# Simpan hasil prediksi dan metrik:
models  = {}
y_preds = {}
metrics = []

# Contoh alur jika melatih 3 varian (Original, SMOTE, RUS):
X_test = df_test.drop(columns=[TARGET_COL]).values
y_test = df_test[TARGET_COL].values

for name, df_train in datasets.items():
    X_train = df_train.drop(columns=[TARGET_COL]).values
    y_train = df_train[TARGET_COL].values

    clf = CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE), ensemble=False)
    clf.fit(X_train, y_train)
    models[name] = clf

    y_pred = clf.predict(X_test)
    y_preds[name] = y_pred

    akurasi = accuracy_score(y_test, y_pred)
    presisi = precision_score(y_test, y_pred, zero_division=0)
    recall  = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"[{name}]: Precision: {presisi:.4f} | Recall: {recall:.4f}")

    # TODO: Tampilkan classification report
    print(f"Classification Report [{name}]:") 
    print(classification_report(y_test, y_pred))

    # TODO: Hitung Confusion Matrix & Elemen TP, TN, FP, FN
    cm = confusion_matrix(y_test, y_pred)
    TN = cm[0][0] * 1.0
    FP = cm[0][1] * 1.0
    FN = cm[1][0] * 1.0
    TP = cm[1][1] * 1.0

    # TODO: Hitung Sensitivity dan Specificity (Formula Modul Hal. 3):
    # Sensitivity (Recall Positif) = TP / (TP + FN) * 100
    # Specificity (Recall Negatif) = TN / (TN + FP) * 100
    sensivity = (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0
    specifiy = (TN / (TN + FP)) * 100 if (TN + FP) > 0 else 0

    metrics.append({
        "Model": name,
        "Akurasi": round(akurasi, 4),
        "Presisi": round(presisi, 4),
        "Recall": round(recall, 4),
        "Sensitivitas": round(sensivity, 4),
        "Specifisitas":  round(specifiy, 4),
        "F1-Score": round(f1, 4),
        "Train Size": len(df_train),
    })


# ─────────────────────────────────────────────────────────────────────────────
# 4. MENYIMPAN TABEL METRIK DAN PREDIKSI DATA UJI
# ─────────────────────────────────────────────────────────────────────────────
df_metrics = pd.DataFrame(metrics)
# Format tabel metrik standar selaras dengan 6-1 dan 6-2
df_metrics_std = pd.DataFrame([
    {
        "Model": m["Model"],
        "Accuracy": m["Akurasi"],
        "Precision": m["Presisi"],
        "Recall": m["Recall"],
        "F1-Score": m["F1-Score"],
        "Train Size": m["Train Size"]
    }
    for m in metrics
])
df_metrics_std.to_csv(output_dir / "model-metrics.csv", index=False)
print(f"\n[OUTPUT] model-metrics.csv tersimpan di: {output_dir / 'model-metrics.csv'}")

df_prediksi = pd.DataFrame({
    "Aktual (y_test)": y_test,
    "Pred_Original": y_preds["Original"],
    "Pred_SMOTE": y_preds["SMOTE"],
    "Pred_RUS": y_preds["RUS"]
})
df_prediksi.to_csv(output_dir / "test-predictions.csv", index=False)
print(f"[OUTPUT] test-predictions.csv tersimpan di: {output_dir / 'test-predictions.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MENYIMPAN MODEL SVM SEBAGAI FILE .pkl (UNTUK DEPLOYMENT FRONTEND)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [col for col in df_test.columns if col != TARGET_COL]
CLASS_LABELS = {0: "Tidak Diabetes", 1: "Diabetes"}

# Parameter skala (mean dan std) diambil dari data training sebelum transformasi
train_clean_path = project_root / "stage-1" / "3-outlier-and-split-data" / "output" / "train-diabetes.csv"
if train_clean_path.exists():
    clean_train_df = pd.read_csv(train_clean_path)
    feature_mean = clean_train_df[FEATURE_COLUMNS].mean()
    feature_std = clean_train_df[FEATURE_COLUMNS].std().replace(0, 1.0)
else:
    raw_df = pd.read_csv(raw_data_path)
    feature_mean = raw_df[FEATURE_COLUMNS].mean()
    feature_std = raw_df[FEATURE_COLUMNS].std().replace(0, 1.0)

# Model terbaik adalah varian RUS (Akurasi: 75.84%, Recall: 76.47%, F1: 68.42%)
best_model_name = "RUS"
best_model = models[best_model_name]

# 1. Simpan model SVM terbaik langsung (.pkl)
svm_model_path = output_dir / "svm_model.pkl"
with open(svm_model_path, "wb") as f:
    pickle.dump(best_model, f)
print(f"\n[OUTPUT] Model SVM terbaik ({best_model_name}) disimpan ke: {svm_model_path}")

# 2. Simpan juga spesifik dengan nama varian
with open(output_dir / "svm_model_rus.pkl", "wb") as f:
    pickle.dump(best_model, f)

# 3. Simpan bundle lengkap (model + parameter scaler + nama fitur)
# Sangat memudahkan integrasi FrontEnd karena input user raw bisa langsung di-transform
svm_bundle = {
    "model": best_model,
    "model_name": f"SVM ({best_model_name})",
    "all_models": models,
    "feature_names": FEATURE_COLUMNS,
    "feature_mean": feature_mean.to_dict(),
    "feature_std": feature_std.to_dict(),
    "class_labels": CLASS_LABELS,
    "metrics": metrics
}
svm_bundle_path = output_dir / "svm_bundle.pkl"
with open(svm_bundle_path, "wb") as f:
    pickle.dump(svm_bundle, f)
print(f"[OUTPUT] Bundle model lengkap (model + scaler) disimpan ke: {svm_bundle_path}")

# Sinkronkan model .pkl ke folder app/ dan final-model/SVM-method/
for target_dir in [project_root / "app", project_root / "final-model" / "SVM-method"]:
    target_dir.mkdir(parents=True, exist_ok=True)
    with open(target_dir / "svm_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(target_dir / "svm_bundle.pkl", "wb") as f:
        pickle.dump(svm_bundle, f)

print(f"[SYNC] Model .pkl berhasil disinkronkan ke:")
print(f"  - {project_root / 'app'}")
print(f"  - {project_root / 'final-model' / 'SVM-method'}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. FUNGSI INFERENSI / PREDIKSI PASIEN TUNGGAL
# ─────────────────────────────────────────────────────────────────────────────
def predict_single_patient(model_name: str, patient_values: dict) -> tuple[int, float, float]:
    """Prediksi satu pasien baru dengan data raw dan transformasi z-score yang konsisten."""
    df_patient = pd.DataFrame([patient_values], columns=FEATURE_COLUMNS)
    df_patient_scaled = (df_patient[FEATURE_COLUMNS] - feature_mean) / feature_std

    model = models[model_name]
    pred = int(model.predict(df_patient_scaled.values)[0])
    proba = model.predict_proba(df_patient_scaled.values)[0]
    return pred, float(proba[0]), float(proba[1])


if __name__ == "__main__":
    # Verifikasi inferensi contoh pasien
    contoh_pasien = {
        "Pregnancies": 2,
        "Glucose": 130,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 110,
        "BMI": 28.4,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 32
    }
    pred, prob_sehat, prob_diab = predict_single_patient("RUS", contoh_pasien)
    print(f"\n[TEST INFERENSI CONTOH]")
    print(f"  Prediksi: {CLASS_LABELS[pred]}")
    print(f"  Probabilitas Tidak Diabetes: {prob_sehat:.2%}")
    print(f"  Probabilitas Diabetes       : {prob_diab:.2%}")

