import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Konfigurasi Path
PROCESSED_DIR = "data/processed"
EVAL_DIR = "data/eval"
CASES_CSV = os.path.join(PROCESSED_DIR, "cases.csv")
TRAIN_CSV = os.path.join(PROCESSED_DIR, "train_cases.csv")
TEST_CSV = os.path.join(PROCESSED_DIR, "test_cases.csv")
QUERIES_JSON = os.path.join(EVAL_DIR, "queries.json")

def split_and_vectorize():
    """
    Memuat data, melakukan splitting (80% Train, 20% Test), dan membangun model TF-IDF
    """
    print("Memuat dataset...")
    df = pd.read_csv(CASES_CSV)
    df['ringkasan_fakta'] = df['ringkasan_fakta'].fillna("")
    df['pasal'] = df['pasal'].fillna("Tidak Diketahui")
    
    # 1. Splitting Data (80:20)
    print("Melakukan Splitting Data (80% Train, 20% Test)...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Simpan hasil split untuk referensi
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    
    # 2. Membangun model TF-IDF dari Data Train (Case Base)
    print("Membangun model TF-IDF dari Data Train...")
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(train_df['ringkasan_fakta'])
    
    return train_df, test_df, vectorizer, tfidf_matrix

def create_queries_from_test(test_df):
    """
    Membuat queries.json dari Data Test.
    """
    if not os.path.exists(EVAL_DIR):
        os.makedirs(EVAL_DIR)
        
    queries = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        query_text = str(row['ringkasan_fakta'])[:200]
        queries.append({
            "query_id": f"q_test_{i+1}",
            "query_text": query_text,
            "ground_truth_label": row['pasal'], # Ground truth yang sebenarnya adalah Pasal
            "original_case_id": row['case_id']
        })
        
    with open(QUERIES_JSON, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=4)
    print(f"[Info] File pengujian awal (queries.json) dibuat dari Data Test: {len(queries)} query")
    return queries

def retrieve(query: str, vectorizer, tfidf_matrix, train_df, k: int = 5):
    """
    Fungsi Retrieval dengan Cosine Similarity
    """
    query_vec = vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Ambil indeks top-k
    top_k_indices = sim_scores.argsort()[-k:][::-1]
    
    results = []
    for idx in top_k_indices:
        case_id = train_df.iloc[idx]['case_id']
        score = sim_scores[idx]
        results.append({"case_id": case_id, "score": score})
        
    return results

def main():
    # 1. Load, Split & Vectorize
    train_df, test_df, vectorizer, tfidf_matrix = split_and_vectorize()
    
    # 2. Buat queries.json dari Data Test
    queries = create_queries_from_test(test_df)
            
    # 3. Pengujian Awal Retrieval
    print("\n--- PENGUJIAN AWAL RETRIEVAL (Mencari di Data Train) ---")
    # Tampilkan 3 query pertama saja sebagai demo
    for q in queries[:3]:
        print(f"\nQuery ID: {q['query_id']}")
        print(f"Teks: {q['query_text']}...")
        
        top_k_results = retrieve(q['query_text'], vectorizer, tfidf_matrix, train_df, k=3)
        
        print("Hasil Pencarian (Top-3 Kemiripan):")
        for i, res in enumerate(top_k_results, 1):
            print(f"  {i}. {res['case_id']} (Score: {res['score']:.4f})")

if __name__ == "__main__":
    main()
