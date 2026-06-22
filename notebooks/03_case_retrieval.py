import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Path
PROCESSED_DIR = "data/processed"
EVAL_DIR = "data/eval"
CASES_CSV = os.path.join(PROCESSED_DIR, "cases.csv")
QUERIES_JSON = os.path.join(EVAL_DIR, "queries.json")

def create_dummy_queries(df, num_queries=5):
    """
    Membuat data pengujian awal (queries.json) secara otomatis 
    dari beberapa data pertama di dataset jika belum ada.
    """
    if not os.path.exists(EVAL_DIR):
        os.makedirs(EVAL_DIR)
        
    queries = []
    for i in range(min(num_queries, len(df))):
        row = df.iloc[i]
        # Kita ambil 150 karakter pertama dari ringkasan_fakta sebagai "query" uji
        query_text = str(row['ringkasan_fakta'])[:150]
        queries.append({
            "query_id": f"q{i+1}",
            "query_text": query_text,
            "ground_truth": row['case_id']
        })
        
    with open(QUERIES_JSON, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=4)
    print(f"[Info] File pengujian awal dibuat di: {QUERIES_JSON}")
    return queries

def load_data_and_vectorize():
    """
    Memuat data CSV dan membangun model representasi vektor TF-IDF
    """
    print("Memuat dataset...")
    df = pd.read_csv(CASES_CSV)
    df['ringkasan_fakta'] = df['ringkasan_fakta'].fillna("")
    
    print("Membangun model TF-IDF...")
    # Menggunakan TF-IDF pada kolom ringkasan_fakta
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['ringkasan_fakta'])
    
    return df, vectorizer, tfidf_matrix

def retrieve(query: str, vectorizer, tfidf_matrix, df, k: int = 5):
    """
    Fungsi Retrieval:
    1. Pre-process query
    2. Hitung vektor query
    3. Hitung cosine-similarity dengan semua case vectors
    4. Kembalikan top-k case_id
    """
    # 1 & 2. Hitung vektor query
    query_vec = vectorizer.transform([query])
    
    # 3. Hitung cosine similarity antara query dengan semua kasus di database
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # 4. Ambil indeks top-k dengan skor kemiripan tertinggi
    top_k_indices = sim_scores.argsort()[-k:][::-1]
    
    results = []
    for idx in top_k_indices:
        case_id = df.iloc[idx]['case_id']
        score = sim_scores[idx]
        results.append({"case_id": case_id, "score": score})
        
    return results

def main():
    # 1. Load data & bangun vektor
    df, vectorizer, tfidf_matrix = load_data_and_vectorize()
    
    # 2. Siapkan data uji (queries.json)
    if not os.path.exists(QUERIES_JSON):
        queries = create_dummy_queries(df)
    else:
        with open(QUERIES_JSON, "r", encoding="utf-8") as f:
            queries = json.load(f)
            
    # 3. Pengujian Awal (Testing fungsi retrieve)
    print("\n--- PENGUJIAN AWAL RETRIEVAL ---")
    for q in queries:
        print(f"\nQuery ID: {q['query_id']}")
        print(f"Teks: {q['query_text']}...")
        print(f"Ground Truth (Seharusnya): {q['ground_truth']}")
        
        # Panggil fungsi retrieve()
        top_k_results = retrieve(q['query_text'], vectorizer, tfidf_matrix, df, k=3)
        
        print("Hasil Pencarian (Top-3):")
        for i, res in enumerate(top_k_results, 1):
            match_status = "✅ MATCH" if res['case_id'] == q['ground_truth'] else "❌"
            print(f"  {i}. {res['case_id']} (Score: {res['score']:.4f}) {match_status}")

if __name__ == "__main__":
    main()
