import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Konfigurasi Path
EVAL_DIR = "data/eval"
RESULTS_DIR = "data/results"
QUERIES_JSON = os.path.join(EVAL_DIR, "queries.json")
PREDICTIONS_CSV = os.path.join(RESULTS_DIR, "predictions.csv")
RETRIEVAL_METRICS = os.path.join(EVAL_DIR, "retrieval_metrics.csv")

def main():
    # Load ground truth dari queries
    with open(QUERIES_JSON, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    # Load prediksi yang dihasilkan di Tahap 4
    if not os.path.exists(PREDICTIONS_CSV):
        print(f"Error: File {PREDICTIONS_CSV} belum ada. Jalankan 04_predict.py terlebih dahulu.")
        return
        
    pred_df = pd.read_csv(PREDICTIONS_CSV)
    
    y_true = []
    y_pred = []
    
    for q in queries:
        q_id = q["query_id"]
        true_label = q["ground_truth_label"]
        y_true.append(true_label)
        
        # Cari prediksi berdasarkan query_id
        pred_row = pred_df[pred_df['query_id'] == q_id]
        if not pred_row.empty:
            predicted_label = pred_row.iloc[0]['predicted_pasal']
            y_pred.append(predicted_label)
        else:
            y_pred.append("Tidak Diketahui")
            
    # Hitung Metrik Evaluasi Klasifikasi (berdasarkan Pasal)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print("--- HASIL EVALUASI MODEL (Prediksi Pasal CBR) ---")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    
    print("\n--- ERROR ANALYSIS ---")
    errors_found = False
    for t, p, q in zip(y_true, y_pred, queries):
        if t != p:
            errors_found = True
            print(f"[!] Query {q['query_id']} GAGAL PREDIKSI PASAL:")
            print(f"    -> Seharusnya : {t}")
            print(f"    -> Terprediksi: {p}")
            
    if not errors_found:
        print("Luar Biasa! Semua query berhasil diprediksi pasalnya dengan benar.")
    else:
        print("\nRekomendasi Perbaikan: Kegagalan disebabkan variasi dokumen. Penambahan dataset Train, eksperimen dengan pembobotan fitur yang lebih baik, atau penggunaan model Transformer disarankan.")

    # Simpan ke CSV
    metrics_data = [{
        "Model": "TF-IDF + Cosine Similarity (Prediksi Pasal)",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1
    }]
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(RETRIEVAL_METRICS, index=False)
    
    print(f"\n[Info] Metrik evaluasi tersimpan di: {RETRIEVAL_METRICS}")

if __name__ == "__main__":
    main()
