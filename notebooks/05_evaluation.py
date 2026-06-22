import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Konfigurasi Path
PROCESSED_DIR = "data/processed"
EVAL_DIR = "data/eval"
CASES_CSV = os.path.join(PROCESSED_DIR, "cases.csv")
QUERIES_JSON = os.path.join(EVAL_DIR, "queries.json")
RETRIEVAL_METRICS = os.path.join(EVAL_DIR, "retrieval_metrics.csv")

def load_data_and_vectorize():
    df = pd.read_csv(CASES_CSV)
    df['ringkasan_fakta'] = df['ringkasan_fakta'].fillna("")
    
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['ringkasan_fakta'])
    return df, vectorizer, tfidf_matrix

def retrieve_top_1(query: str, vectorizer, tfidf_matrix, df):
    """Fungsi pembantu untuk mengambil prediksi Top-1"""
    query_vec = vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = sim_scores.argmax()
    return df.iloc[best_idx]['case_id']

def main():
    df, vectorizer, tfidf_matrix = load_data_and_vectorize()
    
    with open(QUERIES_JSON, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    y_true = []
    y_pred = []
    
    # Kumpulkan ground truth vs prediksi
    for q in queries:
        y_true.append(q["ground_truth"])
        pred_case = retrieve_top_1(q["query_text"], vectorizer, tfidf_matrix, df)
        y_pred.append(pred_case)
        
    # Hitung Metrik menggunakan sklearn.metrics
    # Parameter zero_division=0 untuk menghindari warning jika ada class yang tidak terprediksi
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print("--- HASIL EVALUASI MODEL (TF-IDF Retrieval) ---")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    
    # Analisis Kegagalan (Error Analysis)
    print("\n--- ERROR ANALYSIS ---")
    errors_found = False
    for t, p, q in zip(y_true, y_pred, queries):
        if t != p:
            errors_found = True
            print(f"[!] Query ID {q['query_id']} GAGAL:")
            print(f"    -> Seharusnya : {t}")
            print(f"    -> Terprediksi: {p}")
            
    if not errors_found:
        print("Luar Biasa! Semua query berhasil diprediksi dengan benar (100% Akurat).")
    else:
        print("\nRekomendasi Perbaikan: Model TF-IDF mungkin kesulitan membedakan teks yang mirip atau terlalu singkat. Penggunaan model Transformer (BERT) bisa menjadi solusi di masa depan.")

    # Simpan ke CSV
    metrics_data = [{
        "Model": "TF-IDF + Cosine Similarity",
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
