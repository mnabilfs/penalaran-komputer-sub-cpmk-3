import os
import glob
import re
from pdfminer.high_level import extract_text

# Konfigurasi Path
PDF_DIR = "data/pdf"
RAW_DIR = "data/raw"

def clean_text(text):
    """
    Membersihkan teks dari hasil ekstraksi PDF.
    - Menghapus whitespace berlebih
    - Mengubah ke huruf kecil (lowercase)
    - Menghapus header/footer umum (bisa disesuaikan)
    """
    # Menghapus spasi dan baris baru yang berlebih
    text = re.sub(r'\s+', ' ', text)
    
    # Mengubah ke huruf kecil
    text = text.lower()
    
    # Contoh penghapusan header/footer atau watermark (Bisa disesuaikan dengan isi PDF)
    text = text.replace('direktori putusan mahkamah agung republik indonesia', '')
    text = text.replace('putusan3.mahkamahagung.go.id', '')
    text = text.replace('disclaimer', '')
    
    return text.strip()

def main():
    # Pastikan folder output ada
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
        
    # Ambil semua file PDF
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    
    if len(pdf_files) == 0:
        print("Tidak ada file PDF di folder data/pdf/")
        return
        
    print(f"Ditemukan {len(pdf_files)} file PDF. Mulai ekstraksi...")
    
    # Proses setiap file PDF
    for i, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        print(f"[{i}/{len(pdf_files)}] Memproses {filename}...")
        
        try:
            # 1. Ekstraksi Teks (Konversi PDF -> Plain Text)
            raw_text = extract_text(pdf_path)
            
            # 2. Pembersihan
            cleaned_text = clean_text(raw_text)
            
            # 3. Simpan hasil di folder data/raw/
            out_filename = f"case_{i:03d}.txt"
            out_path = os.path.join(RAW_DIR, out_filename)
            
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
                
        except Exception as e:
            print(f"Gagal memproses {filename}: {e}")
            
    print("\nEkstraksi selesai! Silakan periksa folder data/raw/")

if __name__ == "__main__":
    main()
