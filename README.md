# Klasifikasi-deteksi-anomali - Muhammad Daniel Ilyasa

## Domain Proyek
- Proyek ini berfokus pada prediksi Anomali pengeluaran pribadi dalam transaksi sehari-hari menggunakan pendekatan deep learning, khususnya model LSTM Autoencoder. Dalam era digital saat ini, transaksi keuangan terjadi begitu cepat dan dalam jumlah besar, sehingga sulit bagi individu untuk menyadari adanya pengeluaran tidak wajar secara manual. Deteksi anomali dalam pengeluaran sangat penting untuk membantu pengguna mengelola keuangan dengan lebih bijak, mencegah kebocoran dana, dan mengidentifikasi aktivitas mencurigakan seperti fraud atau pemborosan tidak disadari.

- Masalah pengeluaran tidak wajar ini menjadi perhatian karena dapat berdampak pada stabilitas keuangan individu dan rumah tangga. Menurut data dari Otoritas Jasa Keuangan (OJK, 2023), 61% masyarakat Indonesia mengalami kesulitan dalam mengelola arus kas bulanan secara konsisten. Selain itu, laporan Bank Dunia (2022) menyebutkan bahwa pengeluaran impulsif dan kurangnya literasi keuangan menjadi faktor utama meningkatnya utang konsumtif di kalangan generasi muda. Di sisi lain, studi oleh Tariq et al. (2021) menunjukkan bahwa model LSTM Autoencoder efektif mendeteksi anomali pada transaksi keuangan berbasis deret waktu, dengan tingkat akurasi lebih tinggi dibandingkan metode statistik konvensional. Hal ini menunjukkan bahwa penerapan pendekatan machine learning dalam konteks keuangan pribadi sangat relevan untuk menjawab tantangan nyata yang dihadapi masyarakat saat ini.

---


## Business Understanding

### Problem Statements


- Bagaimana cara mendeteksi transaksi pengeluaran yang tidak wajar dalam riwayat keuangan pribadi?
- Apa metrik dan pendekatan terbaik untuk mengklasifikasikan pengeluaran sebagai normal atau anomali?

### Goals

- Membangun model machine learning yang mampu mempelajari pola pengeluaran pengguna dari waktu ke waktu.
- Menghasilkan klasifikasi pengeluaran: normal atau anomali dengan akurasi tinggi.



    ### Solution statements
    - Menggunakan LSTM Autoencoder sebagai model utama untuk mendeteksi outlier pada data deret waktu pengeluaran.
    - Melakukan fine-tuning terhadap parameter LSTM (jumlah neuron, epochs, learning rate) untuk meningkatkan performa deteksi.
    - Menganalisis fitur yang saling berhubungan  dengan pendekatan melalui visualisasi hasil deteksi untuk interpretasi pengguna

## Data Understanding
Dataset yang digunakan dalam proyek ini diambil dari **Kaggle**: [Daily Transactions Dataset](https://www.kaggle.com/datasets/prasad22/daily-transactions-dataset/data). yang berisi data transaksi harian selama beberapa bulan-tahun. Dataset ini mensimulasikan perilaku transaksi keuangan harian seseorang, dan sangat cocok untuk proyek deteksi anomali karena pola pengeluaran yang tampak alami dapat menjadi dasar untuk mendeteksi outlier atau penyimpangan.

### Jumlah Data
Dataset terdiri dari **2.461 entri** (baris) dan **8 fitur** (kolom), dengan rincian sebagai berikut:

### **Deskripsi Fitur Dataset**

| Nama Kolom       | Tipe Data         | Deskripsi                                                                 |
|------------------|-------------------|---------------------------------------------------------------------------|
| `Date`           | datetime64        | Tanggal transaksi terjadi (format: YYYY-MM-DD)                           |
| `Mode`           | category          | Metode pembayaran, seperti "Cash", "Card", atau "Online"                 |
| `Category`       | category          | Kategori utama transaksi seperti "Food", "Transport", "Salary", dll     |
| `Subcategory`    | category (nullable) | Subkategori transaksi, seperti "Lunch", "Bus", "Bonus", dll            |
| `Note`           | object (nullable) | Catatan tambahan pada transaksi, biasanya berupa teks bebas              |
| `Amount`         | float64           | Jumlah uang yang terlibat dalam transaksi; bisa negatif atau positif     |
| `Income/Expense` | category          | Menandakan apakah transaksi adalah pemasukan (`Income`) atau pengeluaran (`Expense`) |
| `Currency`       | category          | Jenis mata uang yang digunakan, contohnya "INR" atau "USD"              |


### Kondisi Data

1. **Missing Values**:  
   - Kolom `Subcategory` memiliki 635 nilai kosong dari total 2.461 baris.  
   - Kolom `Note` memiliki 521 nilai kosong.  
   - Kolom lain seperti `Date`, `Mode`, `Category`, `Amount`, `Income/Expense`, dan `Currency` **tidak memiliki missing values**.  
   - Penanganan missing value akan dipertimbangkan saat tahap *data preparation*, khususnya untuk kolom `Subcategory` dan `Note`.

2. **Duplikat**:  
   - Setelah dilakukan pengecekan menggunakan `.duplicated()`, **terdapat 11 baris yang duplikat** dalam dataset.  
   - maka dilakukan `drop_duplicates(inplace=True)` untuk menghilangkan duplikat data.

3. **Outlier**:  
   - Nilai ekstrem terdeteksi pada fitur `Amount`, di mana nilai maksimum mencapai **Rp250.000**, jauh di atas nilai rata-rata **Rp2.751** dan median **Rp100**.  
   - Distribusi `Amount` sangat condong ke kanan (right-skewed), menandakan adanya **outlier yang signifikan**.  
   - Outlier ini justru **menjadi fokus utama analisis**, karena proyek ini bertujuan mendeteksi transaksi-transaksi yang menyimpang dari pola umum.
### Uraian Fitur

1. **`Date`**  
   - **Tipe**: `datetime64[ns]`  
   - **Deskripsi**: Tanggal transaksi dilakukan. Fitur ini digunakan untuk analisis waktu dan pola pengeluaran harian.

2. **`Mode`**  
   - **Tipe**: `category`  
   - **Deskripsi**: Metode pembayaran yang digunakan pada transaksi, seperti "Cash", "Credit Card", atau "Bank Transfer".

3. **`Category`**  
   - **Tipe**: `category`  
   - **Deskripsi**: Kategori utama dari transaksi, misalnya "Food", "Utilities", "Salary", dll.

4. **`Subcategory`**  
   - **Tipe**: `category`  
   - **Deskripsi**: Subkategori yang lebih spesifik dari transaksi, seperti "Groceries" dalam kategori "Food". Beberapa nilai pada kolom ini kosong.

5. **`Note`**  
   - **Tipe**: `object`  
   - **Deskripsi**: Catatan tambahan atau deskripsi singkat dari transaksi. Fitur ini bersifat tekstual dan tidak selalu diisi.

6. **`Amount`**  
   - **Tipe**: `float64`  
   - **Deskripsi**: Nilai nominal transaksi. Bisa berupa pengeluaran atau pemasukan tergantung jenis transaksinya.

7. **`Income/Expense`**  
   - **Tipe**: `category`  
   - **Deskripsi**: Menandai apakah transaksi merupakan pemasukan (`Income`) atau pengeluaran (`Expense`).

8. **`Currency`**  
   - **Tipe**: `category`  
   - **Deskripsi**: Mata uang yang digunakan dalam transaksi, seperti "INR" atau lainnya.


### Exploratory Data Analysis
visualisasi pergerakan jumlah Income dan Expense per bulan dari tahun 2015 hingga 2018.
![enter image description here](https://github.com/daniel020901/Klasifikasi-deteksi-anomali/blob/master/Tren_visualisasi.png)




### Data Preparation

Pada tahap ini, data dipersiapkan untuk digunakan dalam proses analisis dan peramalan pengeluaran harian berbasis deep learning. Teknik yang diterapkan dilakukan secara berurutan sebagai berikut:

1. **Pemilihan Fitur Relevan**  
   Dari keseluruhan dataset, hanya tiga kolom yang dipilih untuk keperluan analisis, yaitu:  
   - `Date`: Digunakan sebagai indeks waktu.  
   - `Amount`: Nilai transaksi yang akan dianalisis.  
   - `Income/Expense`: Digunakan untuk memfilter data hanya dengan nilai `Expense`, karena fokus prediksi adalah pada pola pengeluaran.  
   Transaksi dengan nilai `Income` dihapus karena tidak relevan untuk tujuan prediksi pengeluaran.

2. **Konversi Mata Uang**  
   Seluruh transaksi yang menggunakan mata uang `INR` dikonversi ke dalam mata uang `IDR` secara manual. Konversi ini mengacu pada nilai tukar per 20 Mei 2025, yaitu:  
   **1 INR = 191.96 IDR**  
   Langkah ini bertujuan untuk menyeragamkan nilai nominal agar hasil analisis lebih konsisten dengan konteks ekonomi lokal.

3. **Normalisasi Nilai Transaksi (Min-Max Scaling)**  
   Nilai `Amount` kemudian dinormalisasi ke dalam rentang 0 sampai 1 menggunakan teknik **MinMaxScaler** dari Scikit-Learn.  
   Proses normalisasi dilakukan agar model lebih stabil dan cepat saat melakukan proses pelatihan, serta untuk menghindari dominasi fitur tertentu karena skala yang besar.

4. **Pemisahan Data Latih dan Uji**  
   Dataset dibagi menjadi dua bagian:  
   - **80% data** digunakan sebagai **data latih**  
   - **20% data** digunakan sebagai **data uji**  
   Pemisahan dilakukan berdasarkan urutan waktu (chronological split), bukan acak, untuk mempertahankan struktur deret waktu yang valid.

5. **Pembuatan Urutan Deret Waktu (Time Series Sequences)**  
   Data kemudian dikonstruksi ke dalam bentuk urutan berjangka waktu (time-windowed sequences) dengan jendela pengamatan selama **29 hari**.  
   Artinya, model akan menggunakan 29 hari pengeluaran terakhir untuk memprediksi pengeluaran pada hari ke-30.  
   Teknik ini penting untuk memungkinkan model belajar pola historis dalam data deret waktu.

Dengan serangkaian tahap ini, data telah siap untuk digunakan dalam proses pelatihan model prediktif untuk mendeteksi dan memperkirakan pola pengeluaran di masa depan.


## Model Development

### Algoritma yang Digunakan: LSTM Autoencoder

Model yang digunakan adalah **LSTM Autoencoder**, yaitu arsitektur jaringan saraf dalam yang dirancang khusus untuk mengolah data sekuensial atau deret waktu. Model ini bekerja dengan cara merekonstruksi input, sehingga dapat digunakan untuk deteksi anomali pada data pengeluaran harian berdasarkan deviasi antara input dan output yang direkonstruksi.

---

### Cara Kerja:

1. **Encoder (LSTM)**  
   Encoder bertugas mempelajari representasi pola dari data input (dalam hal ini urutan data pengeluaran harian).  
   - Layer pertama LSTM dengan 50 unit dan `return_sequences=True` akan memproses seluruh urutan.
   - Layer kedua LSTM dengan 24 unit mereduksi urutan menjadi representasi fitur yang lebih ringkas.

2. **RepeatVector**  
   Setelah urutan dikompresi menjadi vektor laten, `RepeatVector` digunakan untuk menduplikasi representasi tersebut sesuai panjang urutan awal. Ini adalah langkah penting untuk mempersiapkan data kembali ke bentuk sekuensial pada tahap decoding.

3. **Decoder (LSTM)**  
   Decoder mencoba merekonstruksi urutan asli dari vektor laten.  
   - Layer pertama LSTM dengan 12 unit menghasilkan urutan kembali, yang didistribusikan ke setiap timestep.
   - Output akhir diproses melalui `TimeDistributed(Dense(...))` untuk menghasilkan nilai prediksi pada setiap langkah waktu.

---

### Komponen Regularisasi:

- **Dropout**: Digunakan pada setiap lapisan untuk mencegah overfitting.
- **L2 Regularization**: Diterapkan pada bobot LSTM guna menstabilkan pembelajaran dan menghindari kompleksitas berlebih.

---

### Keunggulan Model:
- **Mengatasi Data Deret Waktu**: Mampu mengenali pola jangka panjang maupun pendek dalam urutan transaksi.
- **Cocok untuk Deteksi Anomali**: Karena output dibandingkan dengan input, selisih yang besar bisa menunjukkan outlier atau anomali keuangan.
- **Arsitektur Simetris**: Encoder dan decoder dirancang secara berimbang agar dapat melakukan rekonstruksi yang optimal.

---


### Inisialisasi Seed:

- `np.random.seed(24)` dan `tf.random.set_seed(24)` digunakan untuk memastikan reprodusibilitas hasil pelatihan model.
  
---

### Parameter Model:

- `LSTM(units=50/24/12)`: Jumlah unit neuron pada masing-masing layer.
- `activation='sigmoid'`: Fungsi aktivasi yang digunakan, cocok untuk data dengan range terbatas.
- `Dropout(rate=0.2 / 0.3)`: Rasio dropout untuk mencegah overfitting.
- `kernel_regularizer=l2(0.001)`: Menambahkan penalti terhadap bobot besar.
- `optimizer=Adam(learning_rate=0.001)`: Optimizer adaptif untuk konvergensi cepat dan stabil.
- `loss='mse'`: Fungsi kerugian Mean Squared Error digunakan karena targetnya adalah merekonstruksi urutan nilai numerik.


---

Dengan arsitektur ini, model dapat mempelajari pola pengeluaran harian dan mengenali ketidakwajaran atau perubahan ekstrem dalam data, yang sangat berguna untuk sistem deteksi anomali berbasis waktu.


## Evaluation
## Evaluation

### Metrik Evaluasi:
Untuk mengevaluasi performa model dalam mendeteksi pengeluaran yang tidak wajar, digunakan metrik:

- **Precision**: Seberapa banyak prediksi anomali yang benar-benar anomali.
- **Recall**: Seberapa banyak anomali yang berhasil terdeteksi oleh model.
- **F1-Score**: Harmoni antara precision dan recall.
- **Confusion Matrix**: Matriks yang menggambarkan distribusi klasifikasi benar dan salah.

Hasil evaluasi model adalah sebagai berikut:

- **Precision**: 1.0000  
- **Recall**: 1.0000  
- **F1-Score**: 1.0000  
- **Confusion Matrix**:<br>
  [[436   0]<br>
  [  0  23]]


Model juga menetapkan **ambang batas deteksi anomali (threshold)** berdasarkan distribusi **Mean Absolute Error (MAE)** dari data training, tepatnya pada **percentile ke-99**, yang menghasilkan nilai threshold: 0.068


---

### Analisis Hasil Evaluasi:

#### 1. **Akurasi Deteksi Transaksi Tidak Wajar**
Model LSTM Autoencoder berhasil mempelajari pola pengeluaran pengguna secara menyeluruh. Hal ini terlihat dari kemampuan model mengklasifikasikan:
- **436 transaksi normal** dengan benar (True Negative),
- **23 transaksi anomali** juga dengan benar (True Positive).

Tanpa ada satupun false positive atau false negative, artinya tidak ada transaksi yang salah klasifikasi. Ini sangat ideal, terutama dalam konteks deteksi keuangan di mana kesalahan deteksi bisa berakibat serius.

#### 2. **Pencapaian Problem Statements**
- **Bagaimana cara mendeteksi transaksi pengeluaran yang tidak wajar dalam riwayat keuangan pribadi?**  
  Model Autoencoder memanfaatkan *reconstruction error* untuk mengukur seberapa besar deviasi suatu transaksi dari pola normal. Jika error melebihi threshold, transaksi tersebut dianggap tidak wajar. Cara ini terbukti efektif dalam kasus ini.

- **Apa metrik dan pendekatan terbaik untuk mengklasifikasikan pengeluaran sebagai normal atau anomali?**  
  Dengan menggunakan metrik Precision, Recall, dan F1-Score, evaluasi performa menjadi lebih menyeluruh. Semua metrik menunjukkan hasil sempurna (1.0), membuktikan bahwa pendekatan LSTM Autoencoder sangat tepat untuk kasus deteksi anomali pada data time-series keuangan.

#### 3. **Pencapaian Goals**
- **Membangun model yang mempelajari pola pengeluaran dari waktu ke waktu**  
  LSTM sebagai arsitektur berbasis deret waktu memungkinkan model memahami urutan kronologis transaksi dan mendeteksi penyimpangan dari pola tersebut.

- **Menghasilkan klasifikasi pengeluaran: normal atau anomali dengan akurasi tinggi**  
  Hasil evaluasi menunjukkan bahwa klasifikasi berjalan sempurna dengan nol kesalahan klasifikasi, yang artinya model sangat andal.

#### 4. **Relevansi dengan Solution Statements**
- **Menggunakan LSTM Autoencoder sebagai model utama**  
  Model ini terbukti unggul dalam mengenali struktur temporal dari data pengeluaran dan mendeteksi penyimpangan yang tak terlihat secara kasat mata.

- **Fine-tuning parameter (jumlah neuron, epochs, learning rate)**  
  Penyesuaian parameter telah dilakukan sehingga menghasilkan model yang tidak hanya akurat, tapi juga stabil dan tidak overfit.

- **Visualisasi hasil deteksi**  
  Histogram dari *train MAE loss* memberi gambaran distribusi error yang memudahkan dalam menetapkan threshold. Ini penting agar pengguna bisa memahami dan memercayai sistem deteksi.

---

### Kesimpulan:
Model berhasil menjawab pertanyaan inti dari proyek ini—yakni **bagaimana cara mendeteksi transaksi yang tidak wajar dengan pendekatan time-series**. Dengan precision dan recall sempurna, serta visualisasi yang intuitif, sistem ini dapat diterapkan secara langsung dalam aplikasi keuangan pribadi sebagai sistem alarm dini terhadap potensi pengeluaran mencurigakan.





**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

