# Sistem Case-Based Reasoning (CBR) Putusan Pengadilan

Repositori ini berisi implementasi sistem *Case-Based Reasoning* (CBR) berbasis Python untuk menganalisis dokumen putusan pengadilan dari Direktori Putusan Mahkamah Agung Republik Indonesia. Proyek ini ditujukan untuk memenuhi tugas mata kuliah Penalaran Komputer.

## Informasi Tim & Dataset

Pengerjaan proyek ini dilakukan oleh 2 mahasiswa:

| Nama Anggota 1 | NIM Anggota 1 | Nama Anggota 2 | NIM Anggota 2 | Jenis Perkara | Nama Pengadilan | Jumlah Putusan |
|---|---|---|---|---|---|---|
| Muhammad Nabil Farras Sulthan | 202310370311259 | Rikza Ahmad Nur Muhammad | 202310370311265 | Pidana Khusus Korupsi | PN Surabaya | 45 |

## Struktur Repositori

Sesuai dengan ketentuan tugas, struktur repositori ini terdiri dari:

```text
├── data/
│   ├── eval/         # Data pengujian awal (queries.json) & hasil metrik evaluasi
│   ├── pdf/          # Kumpulan file PDF putusan asli yang diunduh dari Direktori MA
│   ├── processed/    # Data terstruktur hasil ekstraksi metadata dan fitur (cases.csv)
│   ├── raw/          # File teks mentah (.txt) hasil konversi dari PDF
│   └── results/      # Hasil prediksi dari tahap Solution Reuse (predictions.csv)
├── notebooks/        # Kumpulan script Python untuk tiap tahapan CBR (Tahap 1 s/d 5)
├── .gitignore        # Mengabaikan file yang tidak perlu di-push
├── README.md         # Dokumentasi petunjuk instalasi dan eksekusi
└── requirements.txt  # Daftar library Python yang dibutuhkan
```

## Persyaratan Instalasi

Pastikan Anda telah menginstal **Python 3.x** di sistem Anda.
Untuk menginstal semua *library* pendukung yang dibutuhkan, buka Terminal/Command Prompt di direktori proyek ini dan jalankan perintah berikut:

```bash
pip install -r requirements.txt
```

Library utama yang digunakan:
- `pdfminer.six` (untuk ekstraksi teks dari PDF)
- `pandas` (untuk pengolahan data terstruktur)
- `scikit-learn` (untuk perhitungan TF-IDF, Cosine Similarity, dan Metrik Evaluasi)

## Panduan Eksekusi Pipeline (End-to-End)

Pipeline CBR dalam proyek ini dibagi menjadi 5 tahapan yang harus dieksekusi secara berurutan.

### 1. Tahap Membangun Case Base (Ekstraksi Teks)
Tahap ini membaca seluruh file PDF putusan di folder `data/pdf/`, mengubahnya menjadi teks biasa (*plain text*), melakukan pembersihan, dan menyimpannya di folder `data/raw/`.
```bash
python notebooks\01_membangun_case_base.py
```

### 2. Tahap Case Representation (Ekstraksi Metadata)
Tahap ini mengambil informasi kunci (seperti Nomor Perkara, Nama Terdakwa, Pasal, dan Ringkasan Fakta) dari teks mentah menggunakan *Regular Expression* (Regex) dan menyimpannya dalam bentuk tabel ke `data/processed/cases.csv`.
```bash
python notebooks\02_case_representation.py
```

### 3. Tahap Case Retrieval (Pencarian Kemiripan)
Tahap ini membangun representasi vektor dokumen menggunakan **TF-IDF** dan mencari kemiripan kasus berdasarkan **Cosine Similarity**. Script ini juga akan otomatis membuat data skenario uji di `data/eval/queries.json`.
```bash
python notebooks\03_case_retrieval.py
```

### 4. Tahap Case Solution Reuse (Prediksi Solusi)
Tahap ini menerapkan algoritma *Weighted Similarity* dengan mengambil *Top-5* kasus termirip. Pasal dari putusan paling mirip (*Top-1*) akan ditarik sebagai prediksi solusi untuk query pengujian. Hasilnya disimpan di `data/results/predictions.csv`.
```bash
python notebooks\04_predict.py
```

### 5. Tahap Model Evaluation (Evaluasi Performa)
Tahap ini menguji akurasi keseluruhan sistem retrieval dan prediksi. Metrik yang dihitung mencakup *Accuracy*, *Precision*, *Recall*, dan *F1-Score* menggunakan fungsi standar dari `scikit-learn`. Laporan akhirnya akan disimpan di `data/eval/retrieval_metrics.csv`.
```bash
python notebooks\05_evaluation.py
```

---
**Catatan:** Pastikan memiliki minimal 30 dokumen PDF putusan pengadilan di dalam folder `data/pdf/` sebelum memulai pipeline ini.
