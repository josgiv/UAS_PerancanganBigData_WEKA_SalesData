Baik, ini adalah draf `README.md` yang sangat rinci dan kompleks berdasarkan laporan dan kode yang Anda berikan. Saya mencoba untuk memasukkan sebanyak mungkin detail yang relevan dan menyajikannya dalam format Markdown yang terstruktur.

```markdown
<p align="center">
  <img src="https://www.ubm.ac.id/wp-content/uploads/2020/02/Logo-UBM-Universitas-Bunda-Mulia-Original-PNG.png" alt="Universitas Bunda Mulia Logo" width="150"/>
</p>

<h1 align="center">Laporan Ujian Akhir Semester: Perancangan Big Data</h1>
<p align="center">
  <strong>Analisis Data E-commerce Menggunakan WEKA untuk Segmentasi Produk, Prediksi Rating, dan Pemodelan Harga</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white"/>
  <img alt="Polars" src="https://img.shields.io/badge/Polars-Framework-purple?logo=polars&logoColor=white"/>
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-Library-blueviolet?logo=pandas&logoColor=white"/>
  <img alt="Scikit-learn" src="https://img.shields.io/badge/Scikit_Learn-ML_Library-orange?logo=scikit-learn&logoColor=white"/>
  <img alt="WEKA" src="https://img.shields.io/badge/WEKA-Data_Mining_Tool-brightgreen?logo=weka&logoColor=white"/>
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white"/>
</p>

<p align="center">
  Disusun Oleh: Josia Given Santoso (36230035) <br/>
  Semester 4 - Sains Data <br/>
  Dosen Pengampu: Eko Wahyu Prasetyo. S.T., M.Eng <br/>
  Universitas Bunda Mulia, Jakarta - 2025
</p>

---

## 📜 Daftar Isi

1.  [Pendahuluan](#-pendahuluan)
    *   [Latar Belakang](#latar-belakang)
    *   [Rumusan Masalah](#rumusan-masalah)
    *   [Tujuan Proyek](#tujuan-proyek)
    *   [Batasan Masalah](#batasan-masalah)
2.  [Struktur Direktori Proyek](#-struktur-direktori-proyek)
3.  [Tahapan Proyek](#-tahapan-proyek)
4.  [Persiapan Data (Data Preparation)](#-persiapan-data-data-preparation)
    *   [Lingkungan Pengembangan dan Perpustakaan](#lingkungan-pengembangan-dan-perpustakaan)
    *   [Proses Persiapan Data](#proses-persiapan-data)
        *   [Penggabungan Data (Merging)](#penggabungan-data-merging)
        *   [Pembersihan dan Transformasi Data](#pembersihan-dan-transformasi-data)
        *   [Penanganan Nilai Hilang (Imputation)](#penanganan-nilai-hilang-imputation)
        *   [Konversi Tipe Data](#konversi-tipe-data)
        *   [Deteksi Outlier](#deteksi-outlier)
        *   [Pembuatan File ARFF untuk WEKA](#pembuatan-file-arff-untuk-weka)
    *   [Output Persiapan Data](#output-persiapan-data)
5.  [Analisis Data Mining dengan WEKA](#-analisis-data-mining-dengan-weka)
    *   [Clustering dengan Algoritma SimpleKMeans](#clustering-dengan-algoritma-simplekmeans)
        *   [Konfigurasi Algoritma](#konfigurasi-algoritma-clustering)
        *   [Hasil Clustering dan Interpretasi](#hasil-clustering-dan-interpretasi)
    *   [Klasifikasi dengan Algoritma J48 dan Random Forest](#klasifikasi-dengan-algoritma-j48-dan-random-forest)
        *   [Perbandingan Kinerja Model](#perbandingan-kinerja-model-klasifikasi)
        *   [Confusion Matrix dan Analisis Kesalahan](#confusion-matrix-dan-analisis-kesalahan)
    *   [Regresi dengan Algoritma M5P dan Regresi Linear](#regresi-dengan-algoritma-m5p-dan-regresi-linear)
        *   [Tujuan dan Variabel](#tujuan-dan-variabel-regresi)
        *   [Algoritma yang Digunakan dan Persamaan](#algoritma-yang-digunakan-dan-persamaan)
        *   [Evaluasi dan Perbandingan Kinerja Model](#evaluasi-dan-perbandingan-kinerja-model-regresi)
6.  [Temuan Utama dan Insight Bisnis](#-temuan-utama-dan-insight-bisnis)
7.  [Cara Menjalankan / Reproduksi Hasil](#-cara-menjalankan--reproduksi-hasil)
    *   [Prasyarat](#prasyarat)
    *   [Langkah-langkah Eksekusi](#langkah-langkah-eksekusi)
8.  [Alat dan Teknologi yang Digunakan](#-alat-dan-teknologi-yang-digunakan)
9.  [Kontributor](#-kontributor)
10. [Tautan Penting](#-tautan-penting)

---

## 📝 Pendahuluan

### Latar Belakang
Perkembangan pesat e-commerce menghasilkan volume data yang masif (Big Data), mencakup harga produk, diskon, rating, dan ulasan pelanggan. Analisis data ini menggunakan teknik seperti Data Preparation, Clustering, Klasifikasi, dan Regresi dapat memberikan wawasan mendalam mengenai perilaku pasar, preferensi pelanggan, dan kinerja produk. Aplikasi WEKA menjadi alat bantu powerful untuk eksplorasi dan pemodelan data, memungkinkan identifikasi pola tersembunyi, prediksi tren, dan pengambilan keputusan bisnis yang lebih strategis. Analisis ini krusial bagi manajer toko untuk mengoptimalkan strategi penetapan harga, promosi, inventaris, dan kepuasan pelanggan.

### Rumusan Masalah
Proyek ini bertujuan untuk menjawab pertanyaan-pertanyaan berikut:
1.  Bagaimana melakukan data preparation terhadap dataset produk e-commerce untuk memastikan kualitas dan kesiapan data dalam analisis Big Data?
2.  Bagaimana mengelompokkan (clustering) item produk berdasarkan `offer_price`, `original_price`, dan `off_now` menggunakan algoritma SimpleKMeans untuk mengidentifikasi segmen produk yang berbeda?
3.  Bagaimana membangun model klasifikasi menggunakan algoritma J48 dan Random Forest untuk memprediksi rating produk berdasarkan atribut-atribut tertentu, serta membandingkan performa kedua model?
4.  Bagaimana membangun model regresi menggunakan algoritma Linear Regression dan M5P untuk memprediksi `offer_price` berdasarkan atribut-atribut terkait, serta mengevaluasi dan membandingkan performa kedua model?

### Tujuan Proyek
1.  Melakukan proses data preparation pada dataset yang diberikan agar data siap untuk analisis lebih lanjut.
2.  Menerapkan algoritma SimpleKMeans untuk melakukan clustering pada data produk, mengelompokkan item berdasarkan harga penawaran, harga asli, dan diskon, serta menginterpretasikan karakteristik masing-masing cluster.
3.  Membangun dan membandingkan model klasifikasi (J48, Random Forest) untuk memprediksi rating produk, mengevaluasi performa berdasarkan metrik standar (akurasi, precision, recall, F-measure) dan menganalisis confusion matrix.
4.  Membangun dan membandingkan model regresi (Linear Regression, M5P) untuk memprediksi `offer_price`, serta mengevaluasi performa berdasarkan MAE dan RMSE.

### Batasan Masalah
*   **Dataset**: Terbatas pada tautan yang disediakan: `https://bit.ly/UASPBD2025`.
*   **Aplikasi**: Wajib menggunakan WEKA untuk analisis data mining. Python digunakan untuk data preparation.
*   **Algoritma Clustering**: Hanya SimpleKMeans, dengan jumlah cluster antara 3 hingga 4.
*   **Algoritma Klasifikasi**: Hanya J48 dan Random Forest. Penggunaan atribut `description` dengan `StringToWordVector` bersifat opsional.
*   **Algoritma Regresi**: Hanya Linear Regression dan M5P.
*   **Luaran**: Laporan dan video (minimal 5 menit).
*   **Fokus Analisis Bisnis**: Perubahan harga, diskon tinggi dengan rating buruk, produk dengan banyak review namun harga tetap tinggi, serta visualisasi menggunakan WEKA Visualize Panel.

---

## 📁 Struktur Direktori Proyek

```
.
├── Datasets/
│   ├── Raw_Datasets/              # Data mentah per kategori produk (misal: laptops/, mobiles/)
│   │   ├── laptops/
│   │   ├── mobiles/
│   │   ├── other_electronics/
│   │   │   ├── ac/
│   │   │   ├── fridge/
│   │   │   ├── smartwatch/
│   │   │   ├── washingmachine/
│   │   │   └── waterpurifier/
│   │   ├── peripherals/
│   │   │   ├── earbuds/
│   │   │   ├── hdd/
│   │   │   ├── memory_cards/
│   │   │   ├── monitor/
│   │   │   ├── pen_drives/
│   │   │   ├── power_bank/
│   │   │   ├── ssd/
│   │   │   └── wired_headset/
│   │   ├── tablet/
│   │   └── tvs/
│   └── Full_Datasets/
│       ├── CSV/                   # File gabungan per kategori dalam format CSV (misal: Full_Laptops.csv)
│       ├── MasterData/            # Gabungan seluruh kategori menjadi satu file master
│       │   ├── MasterDataSales.csv
│       │   ├── MasterDataSales.xlsx
│       │   └── category_processing_summary.csv
│       ├── FinalDataCSV/          # Dataset siap pakai (versi nominal dan numerik) setelah cleaning
│       │   ├── FinalDataSales.csv
│       │   └── FinalDataSalesNumeric.csv
│       └── ARFF/                  # Konversi data final ke format ARFF (untuk WEKA)
│           ├── FinalDataSales.arff
│           └── FinalDataSalesNumeric.arff
├── Notebooks/
│   ├── notebook.ipynb             # Jupyter Notebook untuk data merging dan cleaning (output nominal)
│   └── Notebook_Numeric.ipynb     # Jupyter Notebook untuk data cleaning (output numerik)
├── README.md                      # File ini
└── (File lainnya seperti laporan, video, dll.)
```

---

## 🚀 Tahapan Proyek

Proyek ini dilaksanakan melalui beberapa tahapan utama:

1.  **Pengumpulan Data**: Mengunduh dataset dari tautan yang diberikan.
2.  **Persiapan Data (Data Preparation)**:
    *   Menggabungkan seluruh file CSV mentah dari berbagai kategori menjadi satu dataset master.
    *   Melakukan pembersihan data (menghapus duplikasi, kolom tidak relevan).
    *   Melakukan transformasi data (membersihkan format angka, mengkategorikan rating).
    *   Menangani nilai yang hilang (imputasi).
    *   Mengonversi dataset akhir ke format `.ARFF` agar kompatibel dengan WEKA. Dua versi dataset disiapkan: satu dengan atribut kategorikal (nominal) dan satu lagi dengan atribut yang sepenuhnya numerik.
3.  **Analisis Eksploratif Data (Exploratory Data Analysis - EDA)**: Menggunakan visualisasi (boxplot, histogram) untuk mendeteksi outlier dan memahami distribusi data (dilakukan sebagai bagian dari Data Preparation).
4.  **Pemodelan Data Mining dengan WEKA**:
    *   **Clustering**: Menggunakan SimpleKMeans untuk mengelompokkan produk.
    *   **Klasifikasi**: Menggunakan J48 dan Random Forest untuk memprediksi rating produk.
    *   **Regresi**: Menggunakan Linear Regression dan M5P untuk memprediksi harga penawaran (`offer_price`).
5.  **Evaluasi Model**: Mengevaluasi kinerja model menggunakan metrik yang sesuai (WCSS untuk clustering; Akurasi, Precision, Recall, F-Measure, Kappa, MAE, RMSE untuk klasifikasi; Correlation Coefficient, MAE, RMSE untuk regresi).
6.  **Interpretasi Hasil dan Insight Bisnis**: Menganalisis hasil pemodelan untuk mendapatkan wawasan bisnis yang dapat ditindaklanjuti.
7.  **Pelaporan**: Menyusun laporan akhir dan video presentasi.

---

## 🛠️ Persiapan Data (Data Preparation)

Proses persiapan data adalah langkah krusial untuk memastikan kualitas data sebelum analisis. Proses ini dilakukan menggunakan Python dalam lingkungan Jupyter Notebook.

### Lingkungan Pengembangan dan Perpustakaan
*   **Bahasa Pemrograman**: Python 3.12.10
*   **Lingkungan**: Jupyter Notebook
*   **Perpustakaan Utama**:
    *   `Polars`: Untuk membaca dan menggabungkan file CSV besar secara efisien dan paralel.
    *   `Pandas`: Untuk transformasi data tabular, pembersihan, dan ekspor ke format CSV/Excel.
    *   `NumPy`: Untuk operasi numerik dan penanganan nilai `NaN`.
    *   `Scikit-learn`: Untuk imputasi nilai hilang (numerik dan kategorikal) menggunakan `SimpleImputer`.
    *   `Matplotlib`: Untuk visualisasi data, khususnya deteksi outlier melalui boxplot dan histogram.
    *   `os` & `glob`: Untuk iterasi otomatis terhadap struktur folder dan file CSV.
    *   `time`: Untuk pengukuran durasi proses.

### Proses Persiapan Data

Dua jalur output akhir disiapkan: versi nominal (`FinalDataSales.csv` & `.arff`) dan versi numerik (`FinalDataSalesNumeric.csv` & `.arff`).

#### Penggabungan Data (Merging)
Dilakukan menggunakan skrip pada `notebook.ipynb` (bagian `Data Merge`):
1.  **Iterasi File**: Menggunakan `os` dan `glob` untuk iterasi semua file `.csv` dalam subdirektori di `Datasets/Raw_Datasets/`.
2.  **Pembacaan CSV per Kategori**: `polars.read_csv()` digunakan untuk membaca setiap file, dengan penyesuaian tipe data awal ke `Utf8` untuk mencegah kesalahan parsing.
3.  **Penambahan Kolom Kategori**: Kolom `product_category` ditambahkan berdasarkan nama folder asal file.
4.  **Penggabungan per Kategori**: Data dari file-file dalam satu kategori digabungkan menjadi satu file CSV di `Datasets/Full_Datasets/CSV/Full_NamaKategori.csv`.
5.  **Pembuatan Master Data**: Semua file `Full_NamaKategori.csv` digabungkan menjadi `MasterDataSales.csv` dan `MasterDataSales.xlsx` di `Datasets/Full_Datasets/MasterData/`. Polars digunakan untuk penggabungan CSV, dan Pandas (dengan `xlsxwriter` atau `openpyxl`) untuk konversi ke Excel.

```python
# Contoh snippet kode merging dengan Polars (dari notebook.ipynb)
# Membaca dan menggabungkan file CSV per kategori
# df = pl.read_csv(fp, schema_overrides={n: pl.Utf8 for n in cols}, infer_schema_length=0)
# full_df = pl.concat(dataframes_cat, how="diagonal_relaxed")
# full_df = full_df.with_columns(pl.lit(category_name).alias("product_category"))
# full_df.write_csv(csv_out, null_value="NaN")

# Menggabungkan semua kategori menjadi MasterDataSales.csv
# master_df_polars = pl.concat(all_dfs_for_master, how="diagonal_relaxed")
# master_df_polars.write_csv(master_csv_path, null_value="NaN")
```

#### Pembersihan dan Transformasi Data
Dilakukan menggunakan skrip pada `notebook.ipynb` (bagian `Data Cleaning`) setelah `MasterDataSales.csv` terbentuk:
1.  **Input**: `MasterDataSales.csv`.
2.  **Hapus Duplikasi**: `df.drop_duplicates().reset_index(drop=True)`.
3.  **Rename Kolom**: `name` → `product_name`, `rating` → `product_rating`.
4.  **Hapus Kolom Tidak Relevan**: `u_id`, `item_link`, `created_at`. Untuk `FinalDataSalesNumeric.csv`, kolom teks seperti `description` dan `product_name` juga dihapus atau di-encode.
5.  **Pembersihan `discount_percent` (`off_now` di kode)**:
    *   Menghilangkan teks "% off".
    *   Menghapus karakter non-numerik (selain titik).
    *   Mengonversi ke tipe `float`.
    ```python
    # df["off_now"] = (
    #     df["off_now"]
    #     .astype(str)
    #     .str.replace(r"%\s*off", "", regex=True)
    #     .str.replace(r"[^\d\.]", "", regex=True)
    #     .replace("", np.nan)
    #     .astype(float)
    # )
    ```
6.  **Pembersihan `offer_price`, `original_price`**:
    *   Menghapus simbol mata uang dan karakter non-numerik.
    *   Mengonversi ke tipe `float`.
    ```python
    # for col in ["offer_price", "original_price"]:
    #     df[col] = (
    #         df[col]
    #         .astype(str)
    #         .str.replace(r"[^\d\.]", "", regex=True)
    #         .replace("", np.nan)
    #         .astype(float)
    #     )
    ```
7.  **Transformasi `product_rating`**:
    *   Awalnya dikonversi ke numerik (`pd.to_numeric`, `errors='coerce'`).
    *   Kemudian dikategorisasi menjadi "No Rating", "Bad" (<3), "Average" (<4), "Good" (>=4). Untuk dataset numerik, ini bisa diubah menjadi skala ordinal atau melalui label encoding.
    ```python
    # def convert_rating_to_nominal(rating):
    #     if pd.isna(rating) or rating == 0: return "No Rating"
    #     elif rating < 3: return "Bad"
    #     elif rating < 4: return "Average"
    #     else: return "Good"
    # df["product_rating"] = df["product_rating"].apply(convert_rating_to_nominal)
    ```

#### Penanganan Nilai Hilang (Imputation)
*   **Numerik**: Kolom seperti `offer_price`, `original_price`, `off_now`, `total_ratings`, `total_reviews` diimputasi menggunakan nilai rata-rata (mean) dengan `SimpleImputer(strategy="mean")`.
*   **Kategorikal**: Kolom seperti `product_name`, `description`, `product_category`, `product_rating` diimputasi menggunakan nilai modus (most frequent) dengan `SimpleImputer(strategy="most_frequent")`.
    *   Untuk dataset numerik (`FinalDataSalesNumeric.csv`), semua kolom diimputasi dengan mean setelah konversi ke numerik.

```python
# Numerik
# num_imputer = SimpleImputer(strategy="mean")
# df[numeric_cols] = pd.DataFrame(num_imputer.fit_transform(df[numeric_cols]), ...)
# Kategorikal
# cat_imputer = SimpleImputer(strategy="most_frequent")
# df[categorical_cols] = pd.DataFrame(cat_imputer.fit_transform(df[categorical_cols]), ...)
```

#### Konversi Tipe Data
*   `total_ratings`, `total_reviews` dikonversi ke `Int64` (integer yang mendukung NaN Pandas).
*   `description` dikonversi ke `str`.

#### Deteksi Outlier
Visualisasi menggunakan `Matplotlib` untuk kolom numerik:
*   **Boxplots**: Untuk melihat sebaran data dan potensi outlier.
*   **Histograms**: Untuk memahami distribusi frekuensi.
*   **Metode IQR**: Dihitung untuk memberikan batasan bawah dan atas, serta jumlah outlier.

#### Pembuatan File ARFF untuk WEKA
Setelah pembersihan dan imputasi, dataset disimpan sebagai `FinalDataSales.csv` (untuk data nominal/campuran) dan `FinalDataSalesNumeric.csv` (untuk data numerik). Kemudian, skrip `Data Saving (ARFF)` (bagian akhir `notebook.ipynb`) mengonversi file CSV ini ke format `.ARFF`:
1.  **Baca CSV**: `FinalDataSales.csv` dibaca.
2.  **Definisi Atribut**: Tipe data untuk setiap kolom ditentukan (NUMERIC, STRING, atau nominal {nilai1,nilai2,...}). Kolom `description` secara eksplisit dijadikan `STRING`. Kolom dengan banyak nilai unik (>50) dipotong menjadi 50 nilai teratas, sisanya menjadi '?'.
3.  **Escape Values**: Nilai string di-escape (misalnya, `'` menjadi `\'`, `\` menjadi `\\`) dan diapit tanda kutip tunggal. Nilai hilang `np.nan` diganti dengan `?`.
4.  **Penulisan File ARFF**: Header `@RELATION`, `@ATTRIBUTE`, dan `@DATA` ditulis ke file `.arff` di `Datasets/Full_Datasets/ARFF/`.

```python
# Contoh pembuatan header ARFF
# arff_lines = ["@RELATION FinalDataSales\n"]
# for attr_name, attr_type in attributes:
#     if isinstance(attr_type, list): # Nominal
#         enum_vals = ",".join(f"'{v}'" for v in attr_type)
#         arff_lines.append(f"@ATTRIBUTE {attr_name} {{{enum_vals}}}")
#     else: # Numeric atau String
#         arff_lines.append(f"@ATTRIBUTE {attr_name} {attr_type}")
# arff_lines.append("\n@DATA")
# arff_lines.extend([",".join(map(str, row)) for row in data_rows])
```

### Output Persiapan Data
*   `Datasets/Full_Datasets/FinalDataCSV/FinalDataSales.csv`: Dataset bersih dengan atribut campuran (numerik dan nominal).
*   `Datasets/Full_Datasets/FinalDataCSV/FinalDataSalesNumeric.csv`: Dataset bersih dengan semua atribut dikonversi menjadi numerik (beberapa kolom teks mungkin di-drop atau di-encode).
*   `Datasets/Full_Datasets/ARFF/FinalDataSales.arff`: Versi ARFF dari `FinalDataSales.csv`.
*   `Datasets/Full_Datasets/ARFF/FinalDataSalesNumeric.arff`: Versi ARFF dari `FinalDataSalesNumeric.csv`.

---

## 📊 Analisis Data Mining dengan WEKA

Analisis dilakukan menggunakan aplikasi WEKA pada file `.ARFF` yang telah disiapkan.

### Clustering dengan Algoritma SimpleKMeans
*   **Tujuan**: Mengelompokkan produk berdasarkan atribut `offer_price`, `original_price`, dan `off_now`. Atribut `total_ratings` dan `product_rating` dihapus untuk analisis ini.
*   **Dataset**: `FinalDataSales.arff` (atau versi numeriknya dengan kolom yang relevan).

#### Konfigurasi Algoritma (Clustering)
*   **Algoritma**: `weka.clusterers.SimpleKMeans`
*   **Jumlah Cluster (-N)**: 4
*   **Distance Function**: EuclideanDistance
*   **Max Iterasi**: 500
*   **Random Seed**: 10
*   **Missing Values**: Diisi dengan Mean pada tahap data preparation.

#### Hasil Clustering dan Interpretasi
Total Instances: 177.909. Iterasi Konvergen: 15. WCSS: 1837.97.

| Cluster | Jumlah Anggota | Persentase | `offer_price` (mean) | `original_price` (mean) | `off_now` (%) | Interpretasi                                         |
| :------ | :------------- | :--------- | :------------------- | :---------------------- | :------------ | :--------------------------------------------------- |
| 0       | 60.395         | 34%        | Rp 2.629             | Rp 7.150                | 64.96%        | **Produk Diskon Besar dan Murah** (flash sale/clearance) |
| 1       | 65.126         | 37%        | Rp 20.395            | Rp 31.273               | 35.01%        | **Produk Menengah dengan Diskon Moderat**             |
| 2       | 5.815          | 3%         | Rp 160.050           | Rp 201.138              | 19.21%        | **Produk Premium dengan Diskon Rendah** (eksklusif)    |
| 3       | 46.573         | 26%        | Rp 24.864            | Rp 28.009               | 10.03%        | **Produk Standar dengan Diskon Minimal**              |

Visualisasi distribusi hasil clustering (menggunakan WEKA Visualize Panel) menunjukkan segmentasi pasar yang jelas berdasarkan harga dan diskon.

### Klasifikasi dengan Algoritma J48 dan Random Forest
*   **Tujuan**: Memprediksi `product_rating` (target class) berdasarkan atribut lainnya.
*   **Dataset**: `FinalDataSales.arff`. Atribut `description` dapat diubah menggunakan filter `StringToWordVector` jika diinginkan.
*   **Mode Evaluasi**: Menggunakan data pelatihan (evaluate on training data) atau cross-validation (10-fold). Laporan menunjukkan evaluasi pada data pelatihan.

#### Perbandingan Kinerja Model (Klasifikasi)

| Metrik Evaluasi | J48 (%)   | Random Forest (%) |
| :-------------- | :-------- | :---------------- |
| Akurasi         | 98.94     | **99.78**         |
| Precision (avg) | 98.9      | **99.8**          |
| Recall (avg)    | 98.9      | **99.8**          |
| F-Measure (avg) | 98.9      | **99.8**          |
| Kappa           | 0.9825    | **0.9963**        |
| MAE             | 0.0087    | **0.006**         |
| RMSE            | 0.0661    | **0.0379**        |

Random Forest menunjukkan kinerja yang superior dibandingkan J48 di semua metrik.

#### Confusion Matrix dan Analisis Kesalahan
*   **J48**:
    *   Kesalahan klasifikasi lebih tinggi pada kelas "Bad" dan "Average".
    *   Banyak instance kelas "Good" salah diklasifikasikan sebagai "Average" (747 kasus).
    *   Pohon keputusan J48 sangat besar (9801 daun), berpotensi overfitting.
*   **Random Forest**:
    *   Secara signifikan mengurangi kesalahan klasifikasi. Hanya 103 instance "Good" salah menjadi "Average".
*   Kedua model mampu mengenali kelas "No Rating" dengan sempurna.

### Regresi dengan Algoritma M5P dan Regresi Linear
*   **Tujuan**: Memprediksi nilai numerik dari atribut `offer_price`.
*   **Dataset**: `FinalDataSalesNumeric.arff` atau `FinalDataSales.arff` dengan atribut non-numerik yang tidak relevan diabaikan atau dikonversi.
*   **Mode Evaluasi**: 10-fold cross-validation.

#### Tujuan dan Variabel (Regresi)
*   **Variabel Target**: `offer_price`
*   **Variabel Prediktor**: `original_price`, `off_now` (status diskon, biner), `total_ratings`, `product_rating` (jika numerik/ordinal).

#### Algoritma yang Digunakan dan Persamaan
1.  **Linear Regression**: Menghasilkan model linear global.
    Persamaan dari laporan:
    `offer_price = 0.7353 * original_price + -151.7720 * off_now + 0.0043 * total_ratings + -352.7806 * product_rating + 6506.8710`
    *   *Catatan: `off_now` dan `product_rating` di sini mungkin perlu perlakuan khusus (misal, dummy encoding untuk `product_rating` jika kategorikal) agar koefisiennya valid.*
2.  **M5P (Model Tree)**: Menggabungkan pohon keputusan dengan regresi linear di setiap daun.
    *   Parameter: `MinNumInstance: 3`, `BuildRegressionTree: True`.

#### Evaluasi dan Perbandingan Kinerja Model (Regresi)

| Metrik                        | Linear Regression | M5P                 |
| :---------------------------- | :---------------- | :------------------ |
| Correlation Coefficient       | 0.976             | **0.9973**          |
| Mean Absolute Error (MAE)     | 3933.8591         | **1114.1958**       |
| Root Mean Squared Error (RMSE)| 7801.4445         | **2657.7637**       |
| Relative Absolute Error (%)   | 18.0795%          | **5.1207%**         |
| Root Relative Squared Error (%| 21.7622%          | **7.4138%**         |
| Total Instances               | 177,909           | 177,909             |

M5P secara signifikan unggul dalam semua metrik, menunjukkan kemampuannya menangkap pola non-linear dan kompleks dalam data.

---

## 💡 Temuan Utama dan Insight Bisnis

Berdasarkan hasil analisis:

1.  **Pentingnya Data Preparation**: Proses persiapan data yang komprehensif adalah fondasi utama untuk analisis Big Data yang akurat dan reliabel.
2.  **Segmentasi Pasar yang Jelas (Clustering)**:
    *   **Cluster 0 (Diskon Besar, Murah)**: Potensi untuk flash sale, bundling. Perlu diawasi jika rating buruk.
    *   **Cluster 1 (Menengah, Diskon Moderat)**: Target konsumen sensitif harga yang mencari kualitas.
    *   **Cluster 2 (Premium, Diskon Rendah)**: Fokus pada branding dan kualitas, bukan diskon.
    *   **Cluster 3 (Standar, Diskon Minimal)**: Produk dengan margin stabil.
3.  **Prediksi Rating Produk (Klasifikasi)**:
    *   Random Forest (akurasi 99.78%) jauh lebih unggul daripada J48 (98.94%) dalam memprediksi rating produk, lebih robust dan kurang rentan overfitting.
4.  **Pemodelan Harga (Regresi)**:
    *   M5P (Correlation 0.9973, MAE rendah) jauh lebih superior daripada Linear Regression dalam memprediksi `offer_price`, mampu menangkap hubungan non-linear.
5.  **Insight Bisnis Tambahan**:
    *   **Optimalisasi Diskon**: Sesuaikan strategi diskon berdasarkan segmen produk yang teridentifikasi.
    *   **Manajemen Produk Berisiko**: Identifikasi produk dengan diskon tinggi namun rating buruk (sering di Cluster 0). Selidiki akar masalah (kualitas, deskripsi, layanan) daripada terus memberi diskon yang dapat merusak reputasi.
    *   **Potensi Harga Premium**: Produk populer dengan banyak review dan rating bagus memiliki potensi dijual dengan harga lebih tinggi. Pertimbangkan strategi harga premium dan tonjolkan ulasan positif.
    *   **Visualisasi**: Penggunaan WEKA Visualize Panel sangat membantu dalam memahami distribusi data dan hasil model.

---

## ⚙️ Cara Menjalankan / Reproduksi Hasil

### Prasyarat
*   **Python**: Versi 3.10 atau lebih tinggi (proyek menggunakan 3.12.10).
*   **Jupyter Notebook/JupyterLab**: Untuk menjalankan file `.ipynb`.
*   **Perpustakaan Python**:
    *   `polars`
    *   `pandas`
    *   `numpy`
    *   `scikit-learn`
    *   `matplotlib`
    *   `openpyxl` (untuk menulis `.xlsx` dengan Pandas)
    *   `xlsxwriter` (opsional, alternatif untuk menulis `.xlsx`)
    Anda dapat menginstalnya menggunakan pip:
    ```bash
    pip install polars pandas numpy scikit-learn matplotlib openpyxl xlsxwriter jupyterlab
    ```
*   **WEKA**: Versi terbaru (proyek tidak menyebutkan versi spesifik, umumnya versi stabil terbaru baik).

### Langkah-langkah Eksekusi

1.  **Clone Repository**:
    ```bash
    git clone https://github.com/josgiv/UAS_PerancanganBigData_WEKA_SalesData.git
    cd UAS_PerancanganBigData_WEKA_SalesData
    ```

2.  **Unduh Dataset Mentah**:
    Pastikan dataset mentah dari `https://bit.ly/UASPBD2025` telah diunduh dan diekstrak ke dalam direktori `Datasets/Raw_Datasets/` sesuai dengan struktur yang ada (misal, `Datasets/Raw_Datasets/laptops/`, `Datasets/Raw_Datasets/mobiles/`, dst.).

3.  **Jalankan Notebook Persiapan Data**:
    *   Buka Jupyter Notebook atau JupyterLab.
    *   Navigasi ke direktori `Notebooks/`.
    *   Jalankan sel-sel dalam `notebook.ipynb` secara berurutan. Skrip ini akan:
        *   Menggabungkan data mentah (`Data Merge`).
        *   Membersihkan data (`Data Cleaning`).
        *   Menyimpan output ke `Datasets/Full_Datasets/FinalDataCSV/FinalDataSales.csv`.
        *   Mengonversi `FinalDataSales.csv` ke `Datasets/Full_Datasets/ARFF/FinalDataSales.arff` (`Data Saving (ARFF)`).
    *   (Opsional) Jalankan `Notebook_Numeric.ipynb` jika Anda ingin menghasilkan `FinalDataSalesNumeric.csv` dan `FinalDataSalesNumeric.arff` secara terpisah dengan logika yang mungkin sedikit berbeda untuk penanganan fitur numerik.

4.  **Gunakan File ARFF di WEKA**:
    *   Buka aplikasi WEKA.
    *   Pilih "Explorer".
    *   **Untuk Clustering**:
        *   Klik "Open file..." dan pilih `Datasets/Full_Datasets/ARFF/FinalDataSales.arff` (atau `FinalDataSalesNumeric.arff` jika hanya atribut numerik yang relevan).
        *   Pergi ke tab "Cluster".
        *   Klik "Choose" dan pilih `weka.clusterers.SimpleKMeans`.
        *   Konfigurasikan parameter (misal, `numClusters` = 4, `seed` = 10, `distanceFunction` = EuclideanDistance).
        *   Pada "Cluster mode", pilih "Use training set". Hapus centang atribut yang tidak diinginkan (seperti `product_rating`, `total_ratings` jika tidak digunakan).
        *   Klik "Start".
        *   Analisis hasil, termasuk visualisasi cluster.
    *   **Untuk Klasifikasi**:
        *   Klik "Open file..." dan pilih `Datasets/Full_Datasets/ARFF/FinalDataSales.arff`.
        *   Pastikan atribut `product_rating` adalah tipe nominal dan terpilih sebagai `class` (di dropdown atas).
        *   Pergi ke tab "Classify".
        *   Klik "Choose" dan pilih algoritma (misal, `trees.J48` atau `trees.RandomForest`).
        *   Pada "Test options", pilih "Use training data" (sesuai laporan) atau "Cross-validation" (folds=10).
        *   Klik "Start".
        *   Analisis hasil, termasuk confusion matrix dan metrik performa.
    *   **Untuk Regresi**:
        *   Klik "Open file..." dan pilih `Datasets/Full_Datasets/ARFF/FinalDataSalesNumeric.arff` (atau `FinalDataSales.arff` dan pastikan atribut target `offer_price` numerik, dan atribut prediktor relevan).
        *   Pastikan atribut `offer_price` terpilih sebagai `class`.
        *   Pergi ke tab "Classify" (WEKA menggunakan tab yang sama untuk regresi).
        *   Klik "Choose" dan pilih algoritma (misal, `functions.LinearRegression` atau `trees.M5P`).
        *   Pada "Test options", pilih "Cross-validation" (folds=10).
        *   Klik "Start".
        *   Analisis hasil, termasuk koefisien korelasi, MAE, RMSE.

---

## 🛠️ Alat dan Teknologi yang Digunakan

*   **Pemrosesan Data**:
    *   Python 3.12.10
    *   Polars (untuk manipulasi data besar yang efisien)
    *   Pandas (untuk manipulasi data tabular)
    *   NumPy (untuk operasi numerik)
    *   Scikit-learn (untuk imputasi dan preprocessing)
    *   Matplotlib (untuk visualisasi data)
*   **Lingkungan Pengembangan**: Jupyter Notebook
*   **Alat Data Mining**: WEKA (Waikato Environment for Knowledge Analysis)
*   **Manajemen Versi**: Git, GitHub

---

## 🧑‍💻 Kontributor

*   **Nama**: Josia Given Santoso
*   **NIM**: 36230035
*   **Program Studi**: Sains Data
*   **Semester**: 4
*   **Universitas**: Universitas Bunda Mulia

---

## 🔗 Tautan Penting

*   **Source Code & Datasets (GitHub)**: [https://github.com/josgiv/UAS_PerancanganBigData_WEKA_SalesData](https://github.com/josgiv/UAS_PerancanganBigData_WEKA_SalesData)
*   **Video Penjelasan Proyek (Google Drive)**: [https://drive.google.com/drive/folders/11GrpKR-I49BJud8RYYyhKxhOKMoZKFqV](https://drive.google.com/drive/folders/11GrpKR-I49BJud8RYYyhKxhOKMoZKFqV)
*   **Dataset Asli**: [https://bit.ly/UASPBD2025](https://bit.ly/UASPBD2025)

---

<p align="center">
  Proyek ini diselesaikan sebagai bagian dari Ujian Akhir Semester mata kuliah Perancangan Big Data.
</p>
```
