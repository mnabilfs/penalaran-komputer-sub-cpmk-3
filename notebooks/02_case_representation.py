import os
import glob
import re
import pandas as pd

# Konfigurasi Path
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "cases.csv")

def extract_metadata(text):
    """
    Fungsi untuk mengekstrak metadata dari teks putusan menggunakan Regular Expression (Regex).
    Karena format teks putusan bisa beragam, regex ini menggunakan pendekatan heuristik.
    """
    metadata = {
        "no_perkara": "Tidak Ditemukan",
        "tanggal": "Tidak Ditemukan",
        "pasal": "Tidak Ditemukan",
        "pihak": "Tidak Ditemukan",
        "ringkasan_fakta": "Tidak Ditemukan",
        "text_length": len(text) # Feature Engineering: Panjang teks (karakter)
    }
    
    # Ekstraksi Nomor Perkara (Contoh: "nomor 106/pid.sus-tpk/2024/pn sby")
    match_no = re.search(r'nomor\s+([0-9]+/[a-z.-]+/[0-9]+/[a-z\s]+)', text)
    if match_no:
        metadata["no_perkara"] = match_no.group(1).strip()
        
    # Ekstraksi Nama Pihak (Terdakwa)
    match_pihak = re.search(r'nama lengkap\s*:\s*(.*?)(?:tempat|umur|lahir)', text)
    if match_pihak:
        metadata["pihak"] = match_pihak.group(1).strip()
        
    # Ekstraksi Pasal (Misal mencari kata pasal diikuti angka)
    match_pasal = re.search(r'pasal\s+(\d+.*?)(?:undang-undang|kuhp|uuri)', text)
    if match_pasal:
        metadata["pasal"] = match_pasal.group(1).strip()
        
    # Ekstraksi Tanggal (Contoh: "tanggal 12 september 2024")
    match_tgl = re.search(r'tanggal\s+(\d{1,2}\s+[a-z]+\s+\d{4})', text)
    if match_tgl:
        metadata["tanggal"] = match_tgl.group(1).strip()
        
    # Ekstraksi Ringkasan Fakta / Amar Putusan
    # Mengambil teks setelah kata "mengadili:" atau "menjatuhkan pidana" sebagai representasi ringkasan
    match_fakta = re.search(r'(mengadili\s*:.*?|menjatuhkan pidana.*?)(?:\n|$)', text)
    if match_fakta:
        # Potong agar tidak terlalu panjang (max 500 karakter)
        metadata["ringkasan_fakta"] = match_fakta.group(1)[:500].strip() + "..."
    else:
        # Jika tidak ketemu, ambil saja 500 karakter pertama dari teks
        metadata["ringkasan_fakta"] = text[:500] + "..."
        
    return metadata

def main():
    # Pastikan folder output ada
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    txt_files = glob.glob(os.path.join(RAW_DIR, "*.txt"))
    
    if len(txt_files) == 0:
        print("Tidak ada file teks di folder data/raw/")
        return
        
    print(f"Memulai ekstraksi metadata untuk {len(txt_files)} dokumen...")
    
    data_records = []
    
    for i, txt_path in enumerate(txt_files, 1):
        filename = os.path.basename(txt_path)
        case_id = filename.replace(".txt", "") # case_001, dll
        
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # Ekstraksi metadata
        metadata = extract_metadata(text)
        
        # Susun data per baris
        record = {
            "case_id": case_id,
            "no_perkara": metadata["no_perkara"],
            "tanggal": metadata["tanggal"],
            "pihak": metadata["pihak"],
            "pasal": metadata["pasal"],
            "ringkasan_fakta": metadata["ringkasan_fakta"],
            "text_length": metadata["text_length"],
            "text_full": text
        }
        
        data_records.append(record)
        print(f"[{i}/{len(txt_files)}] {case_id} selesai.")
        
    # Simpan ke CSV dengan pandas
    df = pd.DataFrame(data_records)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    
    print(f"\nEkstraksi metadata selesai! Data tersimpan di: {OUTPUT_CSV}")
    print("\nContoh 3 data teratas:")
    print(df[['case_id', 'no_perkara', 'pihak']].head(3))

if __name__ == "__main__":
    main()
