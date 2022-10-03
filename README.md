# Laporan Proyek Machine Learning - Bima Surya Nurwahid

## Domain Proyek

Domain yang dipilih dalam proyek machine learning ini adalah Investment & financial, dengan judul **Predictive Analytics of Tesla stock**

### Latar Belakang 
Pada masa sekarang dari kalangan yang tua maupun yang muda sudah tak asing lagi dengan investasi, Entah investasi properti, Saham, Stock, maupun Crypto. Orang atau investor yang sudah memulai perjalanan invest nya sejak dini kebanyakan mereka mempunyai trading plan atau planning keuangan mereka untuk kedepannya.

Orang yang berinvestasi pada sejak dini itu merupakan salah satu orang yang sudah peduli tentang masa depanya entah itu investasi dalam banyak hal properti ataupun saham misalnya, beberapa tahun belakangan ini kususnya saham teknologi company, hampir bisa menjamin hari tua seseorang jika dia berinvestasi pada company yang tepat, contohnya **Tesla** kemarin yang melonjak naik ratusan persen dikarenakan teknologi yang ia munculkan terlebih lagi **Twitter** sudah diakuisisi menjadi pemilik penuh dari seorang **Elon Musk** sekaligus pemilik Tesla. Oleh karena itu proyek ini akan mempermudah para investor muda untuk berinvestasi lebih pintar pada company yang tidak akan merugikan dalam rentan waktu beberapa tahun kedepan menggunakan Machine Learning untuk memprediksi kemungkinan naik turunnya harga di pasar kedepannya.

Forecasting merupakan bagian dari Machine Learning dimana itu salah teknik yang dapat meramalkan keadaan, harga dimasa yang akan datang menggunakan data data historynya. hal ini masih termasuk kedalam Time Series forecasting, dengan mendeteksi pola dan kecenderungan data historinya yang kemudian diformulasikan kedalam model machine learnign yang nanti akan kita gunakan untuk meprediksi data yang akan datang beberapa tahun kedepan.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang yang telah dijelaskan diawal, berikut beberapa permasalahan yang dapat diselesaikan dalam proyek ini :
- Apakah Tesla adalah jawaban untuk para investor bahwa Tesla adalah market yang bagus untuk kedepannya?
- Bagaimana cara membangun model machine learning untuk memprediksi harga stock Tesla kedepanya?

### Goals
Tujuan dibuatnya proyek ini sebagai berikut :
- Tesla menjadi jawaban untuk para investor dalam berinvestasi jangka panjang dan harga market dapat dianalisis menggunakan machine learning.
- Membangun model machine learning untuk memprediksi harga stock Tesla dengan tingkat akurasi yang tinggi.


### Solution statements
Solusi yang bisa dilakukan agar goals dapat terpenuhi sebagai berikut :
* Melakukan analisa dan eksplorasi lebih jauh pada dataset dan memvisualisasikanya agar mendapat gambaran yang kuat, seperti:
  - Melakukan pembagian dataset
  - Mencari korelasi pada dataset untuk mencari dimana variabel dependent dan variabel  independent. 
  - Melakukan Normalization pada dataset terutama pada fitur numerik.
  - Membuat model regresi guna meprediksi bilangan kontinu harga saham dimasa yang akan datang.
  - Menangani jika terjadinya missing value pada data.
  - Jika terdapat outliner, menganganinya dengan metode IQR.

* Berikut merupakan list algortima yang dicoba dalam model:
  - Support Vector Machine (Support Vector Regression)
  - K-Nearest Neighbors (KNN)
  - Boosting Algorithm (Gradient Boosting Regression)


## Data Understanding
Pada proyek ini saya mengambil dataset publik dari Kaggle yang berjudul _Tesla Stock Price_ (https://www.kaggle.com/datasets/rpaguirre/tesla-stock-price).

Dataset yang digunakan memiliki format .csv, mempunyai total 4431 data dengan 7 kolom diantaranya (Date, Open, High, Low, Close, Adj Close dan Volume), berikut merupakan penjelasan masing masing kolom:
- **Date**: Opening rekap data
- **High** : Highest price per day
- **Low** : Lowest price per day
- **Open** : Opening price per day
- **Close** : Closing price per day
- **Adj Close** : Closing price per day after counting stock split or stock reverse
- **Volume** : Volume Transaction price per day

### Eksploratory Data Analysis
sebelum beranjak ke Data Preparation, kita harus mengetahui data, seperti korelasi, outliner, dan analisis Univariate dan Multivariate anailisis
- Mengangani adanya Outliner 

![outliner_before](https://user-images.githubusercontent.com/105061172/193448036-ec730c1b-88c7-488d-90bb-f8ff1fbbb64a.jpeg)

Gambar 1.Outliner Dataset

Terlihat jika di atas banyak terdapat outlier pada setiap variabel, lalu untuk mengatasinya nantinya penulis akan menerapkan batas bawah dan batas atas menggunakan metode IQR

- Unvariate Analysis

![unvariative](https://user-images.githubusercontent.com/105061172/193448172-937d1d3c-ca01-4172-8e11-69ac6799f2aa.jpeg)


Gambar 2.Unvariate Analysis

Pada Gambar 2, Karena yang kita cari adalah Adj Close, maka kita akan fokus ke salah satu kolom dimana kolomnya adalah kolom Adj Close dan Terlihat pada grafik bahwa semua data cenderung distribusi nilainya miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model nantinya.

- Multivariate Analysis

Pada kali ini kita akan menganalisis korealsi Adj Close terhadap fitur lain. dan dapat disimpulkan bahwa Adj Close memiliki korelasi positif yang kuat terhadap kolom ***Open, High, Low  dan  Close***, sedangkan terhadap kolom Volume tidak memiliki korelasi yang kuat.

![multivariate](https://user-images.githubusercontent.com/105061172/193504255-ee928f24-b5a0-4df0-99fb-1cd8a7af4654.jpeg)


Gambar 3.Multivariate Analysis

Pada Gambar 3, kolom Adj Close memiliki korelasi positif terhadap kolom **High, Open, Low, dan Close**. Dan kolom **Volume** memiliki korelasi yang lemah. 

Selanjutnya kita bisa membuat heatmap korelasi pada data menggunakan library Seaborn

![heatmap_correlation](https://user-images.githubusercontent.com/105061172/193504286-7395bc40-a7a2-42c0-a014-1c60561b3ebd.jpeg)


Gambar 4. Korelasi Heatmap

Pada Gambar 4, Terlihat pada matriks korelasi di atas dapat disimpulkan bahwa semua variabel memiliki keterikatan dan korelasi yang kuat antar variabel lainnya

## Data Preparation
Pada tahap ini saya melakukan beberapa tahapan dalam pemrosesan data:

### Melakukan Penanganan Missing Value dan Outliner
Tahapan awal adalah menghilangkan Missing value pada dataset yang memiliki 2 cara untuk dihapus atau akan diisi dengan nilai rata rata menggunakan library Simpleimputer, karena dataset yang saya gunakan tidak memilki missing value kita bisa lanjut ke tahap selanjutnya, Dan untuk mengatasi outlier pada proyek, penulis menggunakan penentuan batas atas dan bawah nilai kuartil pada data dengan menggunakan metode IQR.

### Splitting atau pembagian dataset
Untuk mengetahui kinerja model ketika dihadapkan pada data yang belum pernah dilihat sebelumnya, maka perlu dilakukan pembagian dataset. Pada proyek ini dataset dibagi menjadi data latih dan data uji dengan rasio 80% untuk data latih dan 20% untuk data uji. Data latih merupakan data yang akan kita latih untuk membangun model machine learning, sedangkan data uji merupakan data yang belum pernah dilihat oleh model dan digunakan untuk melihat kinerja atau performa dari model yang dilatih. Pembagian dataset dilakukan dengan modul train_test_split dari scikit-learn.

### Menghapus fitur yang tidak diperlukan 
setelah diolah ternyata kita hanya memerlukan kolom **Open, High, Low, dan Adj Close**. oleh karena itu kita bisa drop atau menghapus fitur selain kolom diatas seperti **Volume, Date dan Close.**


## Modeling
Pada tahap ini kita menggunakan 3 buah algoritma diantaranya ada _Support Vector Regression, Gradient Boost dan KNN_.

### Support Vector Regression 

Algoritma ini hampir sama seperti SVM tetapi pada SVM biasa digunakan dalam klasifikasi. Pada SVM, algoritma tersebut berusaha mencari jalan terbesar yang bisa memisahkan sampel dari kelas berbeda, sedangkan SVR mencari jalan yang dapat menampung sebanyak mungkin sampel di jalan. Berikut merupakan Hyper Parameter yang digunakan dalam model: 
 - kernel : Hyperparameter ini biasa digunakan untuk menghitung kernel pada matriks sebelumnya, pada model ini menggunakan kernel **"rbf"** dikarenakan konsep dari kernel ini yang paling banyak digunakan dalam klasifikasi data yang tidak dappat dipisahkan secara linier. 
 - C : Hyperparameter ini biasa digunakan untuk menukar klasifikasi yang benar dari contoh training terhadap maksimalisasi margin fungsi keputusan, pada model ini kita gunakan nilai **1000**.
 - gamma : Hyperparameter ini biasa digunakan untk menetukan seberapa jauh pengaruh satu contoh pelatihan mencapai, dengan nilai rendah berarti jauh dan nilai tinggi berarti dekat, dalam model ini kita berikan nilai gamma **0.003**.

#### kelebihan
- Lebih efektif pada data dimensi tinggi (data dengan jumlah fitur yang banyak)
- Memori lebih efisien karena menggunakan subset poin pelatihan

#### Kekurangan 
- Sulit dipakai pada data skala besar

### Gradient Boost

Gradient Boosting adalah algoritma machine learning yang menggunakan teknik ensembel learning dari decision tree untuk memprediksi nilai. Gradient Boosting sangat mampu menangani pattern yang kompleks dan data ketika linear model tidak dapat menangani. Untuk hyperparameter yang digunakan pada model ini ada 3 yaitu: 
- learning_rate : Hyperparameter training yang digunakan untuk menghitung nilai koreksi bobot padded pada waktu proses training. Umumnya nilai learning rate berkisar antara **0** hingga **1**, dalam fitting model ini menggunakan nilai **0.01**.
- n_estimators : Jumlah tahapan boosting yang akan dilakukan pada model, pada model ini menggunakan nilai **1000** tahapan.
- criterion : Hyperparameter yang biasanya digunakan untuk menemukan fitur dan ambang batas optimal dalam membagi data, pada model ini menggunakan "**squared_error**" dimana untuk kesalahan kuadrat rata rata.

#### kelebihan
- Hasil pemodelan yang lebih akurat
- Model yang stabil dan lebih kuat (robust)
- Dapat digunakan untuk menangkap hubungan linear maupun non linear pada data

#### Kekurangan 
- Pengurangan kemampuan interpretasi model
- Waktu komputasi dan desain tinggi
- Tingkat kesulitan yang tinggi dalam pemilihan model

### K-Nearest Neighbors (KNN)

K-Nearest Neighbors merupakan algoritma machine learning yang bekerja dengan mengklasifikasikan data baru atau membandingkan antara sampel dengan sampel yang lain dengan menggunakan kemiripan dengan tetangganya atau bisa dikatakan antara data baru dengan sejumlah data (k) pada data yang telah ada. Algoritma ini dapat digunakan untuk klasifikasi dan regresi guna menghasilkan persamaan yang dihasilkan akan lebih halus. Untuk hyperparameter yang digunakan pada model ini hanya 1 yaitu :

#### kelebihan
- Sangat efektif apabila jumlah datanya banyak
- Sederhana dan Mudah diimplementasikan

#### Kekurangan 
- Sensitif pada outlier dan lebih lambat secara signifikan

Dapat disimpulkan model terbaik yang digunakan untuk dataset ini ialah model KNN di mana KNN memiliki nilai akurasi yang tinggi ketimbang kedua model lainnya.

## Evaluation
Pada tahap evaluasi ini metrik yang digunakan adalah Mean Squared Error (MSE), dimana dia akan mengukur seberapa dekat garis pas dengan titik pada data dan mengukur kinerja pada model. Dan ini menunjukkan bahwa KNN merupakan model terbaik dalam memprediksi harga stock Tesla kedepannya.

![mse_rumus](https://user-images.githubusercontent.com/73319544/191768488-1d350af9-cb15-4fe1-8cd5-7c8ded16aaf9.png)

Gambar 5. Rumus MSE

Keterangan :
- n = Jumlah titik data
- Yi = nilai sesungguhnya 
- Yi_hat = nilai prediksi

berikut merupakan Visualisai dari hasil akurasi model :

![mse](https://user-images.githubusercontent.com/105061172/193508315-d0d6e65f-0bed-474f-9823-4db40229a8ec.jpeg)


Gambar 6. Hasil MSE dari model

Pada Gambar 6, Kita bisa lihat hasil dari MSE model, dimana semakin kecil MSE yang diperoleh oleh model maka akan semakin optimal algortima tersebut.

![mse_plot](https://user-images.githubusercontent.com/105061172/193509304-9027691a-9b67-441c-8eaa-0bed1a5fb6b4.jpeg)


Gambar 7. Plot Visualisasi MSE pada model

Pada Gambar 7, kita bisa melihat hasil plot dari model terhadap beberapa algortma yang digunakan, ada 2 algoritma yang sangat cocok untuk gunakan diantaranya ada KNN dan Gradient Boosting.

![mse_accuracy](https://user-images.githubusercontent.com/105061172/193509258-cca133cf-6e96-40cb-b18a-95968d848f4d.jpeg)





Gambar 8. Hasil akurasi MSE model

Pada Gambar 8, Kita bisa melihat akurasi dari setiap algortima yagn digunakan dan kita bisa simpulkan bahwa KNN atau K-Nearest Neighbors merupakan algortma paling optimal untuk model. 

Pada proyek ini semua model berjalan dengan sangat baik dan maksimal dan hanya terdapat selisih sangat kecil diantara ketiganya akan tetapi kita akan memilih model yang paling tinggi akurasinya, dimana K-Nearest Neighbors (KNN) adalah algortima yang memiliki nilai tertinggi.

### Forecasting
pada tahap ini saya akan mencoba memprediksi menggunakan algortma yang kita pili diatas yaitu KNN dalam kurun waktu 30 hari kedepan 

![Prediksi](https://user-images.githubusercontent.com/105061172/193508929-52ddc585-6831-41a5-948d-2db83f5dc051.jpeg)

 Bisa kita lihat prediksi harga yang akan datang dalam kurun waktu 30 hari kedepan yang sudah diprediksi menggunakan KNN yang telah kita pilih sebagai algortima yang paling optimal.

