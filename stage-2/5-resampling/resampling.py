from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# Menentukan nama kolom target dan seed agar hasil resampling dapat diulang.
TARGET_COL = "Outcome"
RANDOM_STATE = 42

# Menentukan lokasi file input berdasarkan lokasi skrip, bukan direktori kerja terminal.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[1]
input_path = (
	project_root
	/ "stage-2"
	/ "4-data-transformation"
	/ "output"
	/ "train-transformed.csv"
)
output_dir = script_dir / "output"
# Membuat folder output jika belum tersedia.
output_dir.mkdir(parents=True, exist_ok=True)

# Membaca data train hasil transformasi dari tahap sebelumnya.
df_train = pd.read_csv(input_path)
# Memastikan kolom target tersedia sebelum data diproses lebih lanjut.
if TARGET_COL not in df_train.columns:
	raise ValueError(f"Kolom target '{TARGET_COL}' tidak ditemukan pada {input_path}.")

# Memisahkan fitur prediktor dari target agar resampling hanya berdasarkan fitur dan label.
X = df_train.drop(columns=TARGET_COL)
y = df_train[TARGET_COL]
# Menghitung jumlah data tiap kelas untuk menganalisis ketimpangan target.
class_counts = y.value_counts().sort_index()
# Memastikan dataset berisi kelas Outcome 0 dan 1 sesuai kebutuhan tugas.
if len(class_counts) != 2 or set(class_counts.index) != {0, 1}:
	raise ValueError("Kolom Outcome harus memiliki dua kelas dengan nilai 0 dan 1.")

# SMOTE memerlukan sedikitnya dua contoh pada kelas yang jumlahnya paling sedikit.
minority_count = int(class_counts.min())
if minority_count < 2:
	raise ValueError("SMOTE membutuhkan minimal dua baris pada kelas minoritas.")

# Mengukur rasio kelas dan menampilkan jumlah serta persentase sebelum resampling.
imbalance_ratio = class_counts.max() / class_counts.min()
print("=== Analisis distribusi target sebelum resampling ===")
print(f"Jumlah data train: {len(df_train)}")
print(f"Outcome 0 (false): {class_counts[0]} ({class_counts[0] / len(y):.2%})")
print(f"Outcome 1 (true) : {class_counts[1]} ({class_counts[1] / len(y):.2%})")
print(f"Rasio mayoritas/minoritas: {imbalance_ratio:.2f}:1")
print(
	"Interpretasi: distribusi kelas tidak sangat timpang, tetapi kelas mayoritas "
	"tetap lebih banyak. SMOTE menambah sampel sintetis kelas minoritas, sedangkan "
	"RUS mengurangi sampel kelas mayoritas."
)

# Memilih jumlah tetangga SMOTE yang tidak melebihi jumlah data kelas minoritas.
smote_neighbors = min(5, minority_count - 1)
# SMOTE menambah data sintetis pada kelas minoritas sampai jumlah kedua kelas setara.
X_smote, y_smote = SMOTE(
	random_state=RANDOM_STATE,
	k_neighbors=smote_neighbors,
).fit_resample(X, y)

# RUS mengambil sebagian kelas mayoritas secara acak agar jumlah kelas menjadi setara.
X_rus, y_rus = RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(X, y)

# Menggabungkan kembali fitur dan target serta menjaga urutan kolom seperti data input.
df_smote = pd.DataFrame(X_smote, columns=X.columns)
df_smote[TARGET_COL] = y_smote
df_smote = df_smote[df_train.columns]

df_rus = pd.DataFrame(X_rus, columns=X.columns)
df_rus[TARGET_COL] = y_rus
df_rus = df_rus[df_train.columns]

# Menyimpan dataset hasil oversampling dan undersampling sebagai CSV.
df_smote.to_csv(output_dir / "train-oversampling-smote.csv", index=False)
df_rus.to_csv(output_dir / "train-undersampling-rus.csv", index=False)

# Menyiapkan jumlah kelas sebelum resampling dan setelah masing-masing metode.
datasets = {
	"Sebelum": y.value_counts().sort_index(),
	"SMOTE": y_smote.value_counts().sort_index(),
	"RUS": y_rus.value_counts().sort_index(),
}
# Membuat tabel ringkasan jumlah dan proporsi tiap kelas untuk dianalisis kembali.
summary_rows = []
for method, counts in datasets.items():
	total = int(counts.sum())
	for outcome in (0, 1):
		summary_rows.append(
			{
				"Metode": method,
				"Outcome": outcome,
				"Kategori": "false" if outcome == 0 else "true",
				"Jumlah": int(counts[outcome]),
				"Persentase": counts[outcome] / total,
			}
		)
summary = pd.DataFrame(summary_rows)
summary.to_csv(output_dir / "resampling-summary.csv", index=False)

# Membuat dua grafik berdampingan: jumlah data dan persentase tiap kelas.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Perbandingan Distribusi Kelas Sebelum dan Sesudah Resampling")
method_names = list(datasets)
positions = range(len(method_names))
bar_width = 0.34
colors = {0: "#3977a8", 1: "#e07a3f"}

# Menggambar batang untuk kedua kelas pada setiap metode resampling.
for outcome, offset in ((0, -bar_width / 2), (1, bar_width / 2)):
	counts = [int(datasets[name][outcome]) for name in method_names]
	percentages = [
		datasets[name][outcome] / datasets[name].sum() * 100
		for name in method_names
	]
	count_bars = axes[0].bar(
		[position + offset for position in positions],
		counts,
		bar_width,
		color=colors[outcome],
		label=f"Outcome {outcome} ({'false' if outcome == 0 else 'true'})",
	)
	axes[1].bar(
		[position + offset for position in positions],
		percentages,
		bar_width,
		color=colors[outcome],
		label=f"Outcome {outcome} ({'false' if outcome == 0 else 'true'})",
	)
	for bar, count in zip(count_bars, counts):
		axes[0].annotate(
			str(count),
			(bar.get_x() + bar.get_width() / 2, bar.get_height()),
			xytext=(0, 3),
			textcoords="offset points",
			ha="center",
			fontsize=9,
		)
	for position, percentage in zip(positions, percentages):
		axes[1].annotate(
			f"{percentage:.1f}%",
			(position + offset, percentage),
			xytext=(0, 3),
			textcoords="offset points",
			ha="center",
			fontsize=9,
		)

# Menambahkan label metode, legenda, garis bantu, dan judul sumbu grafik.
for axis in axes:
	axis.set_xticks(list(positions), method_names)
	axis.legend()
	axis.grid(axis="y", linestyle="--", alpha=0.35)

axes[0].set_title("Jumlah Sampel")
axes[0].set_ylabel("Jumlah baris")
axes[1].set_title("Proporsi Kelas")
axes[1].set_ylabel("Persentase")
axes[1].set_ylim(0, 65)
fig.tight_layout()

# Menyimpan grafik perbandingan lalu menutup figure agar tidak menumpuk di memori.
fig.savefig(output_dir / "resampling-comparison.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# Menampilkan distribusi hasil serta perbedaan jumlah data yang dibuat/dihapus.
print("\n=== Hasil resampling ===")
for method, counts in datasets.items():
	print(
		f"{method:8} | Outcome 0: {counts[0]} | Outcome 1: {counts[1]} "
		f"| Total: {counts.sum()}"
	)
print("\nAnalisis perbedaan:")
print(
	f"- SMOTE menyeimbangkan kelas dengan menambah "
	f"{len(df_smote) - len(df_train)} baris sintetis; seluruh data asli dipertahankan."
)
print(
	f"- RUS menyeimbangkan kelas dengan mengurangi "
	f"{len(df_train) - len(df_rus)} baris dari kelas mayoritas; sebagian data dibuang."
)
print(f"\nCSV dan visualisasi tersimpan di: {output_dir}")
