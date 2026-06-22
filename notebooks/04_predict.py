import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Path
PROCESSED_DIR = "data/processed"
EVAL_DIR = "data/eval"
RESULTS_DIR = "data/results"
CASES_CSV = os.path.join(PROCESSED_DIR, "cases.csv")
QUERIES_JSON = os.path.join(EVAL_DIR, "queries.json")
PREDICTIONS_CSV = os.path.join(RESULTS_DIR, "predictions.csv")

def load_data_and_vectorize():
    df = pd.read_csv(CASES_CSV)
    df['ringkasan_fakta'] = df['ringkasan_fakta'].fillna("")
    df['pasal'] = df['pasal'].fillna("Tidak Diketahui")
    
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['ringkasan_fakta'])
    
    # Buat dictionary solusi (case_id -> solusi)
    # Di sini kita anggap "solusi" adalah gabungan Pasal dan Ringkasan
    case_solutions = {}
    for _, row in df.iterrows():
        case_solutions[row['case_id']] = {
            "pasal": row['pasal'],
            "teks_putusan": row['ringkasan_fakta']
        }
        
    return df, vectorizer, tfidf_matrix, case_solutions

def retrieve(query: str, vectorizer, tfidf_matrix, df, k: int = 5):
    query_vec = vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_k_indices = sim_scores.argsort()[-k:][::-1]
    
    results = []
    for idx in top_k_indices:
        results.append({
            "case_id": df.iloc[idx]['case_id'], 
            "score": sim_scores[idx]
        })
    return results

def predict_outcome(query: str, vectorizer, tfidf_matrix, df, case_solutions) -> tuple:
    # Ambil top 5 kasus termirip
    top_k = retrieve(query, vectorizer, tfidf_matrix, df, k=5)
    
    # Pendekatan Weighted Similarity untuk memprediksi "Pasal" yang tepat
    pasal_scores = {}
    for case in top_k:
        c_id = case["case_id"]
        score = case["score"]
        pasal = case_solutions[c_id]["pasal"]
        
        if pasal not in pasal_scores:
            pasal_scores[pasal] = 0
        pasal_scores[pasal] += score # Menjumlahkan bobot kemiripan
        
    # Cari pasal dengan bobot tertinggi
    predicted_pasal = max(pasal_scores, key=pasal_scores.get)
    
    # Ambil amar putusan/ringkasan dari kasus termirip pertama sebagai referensi solusi
    best_case_id = top_k[0]["case_id"]
    predicted_text = case_solutions[best_case_id]["teks_putusan"]
    
    # Gabungkan menjadi predicted solution
    predicted_solution = f"Prediksi Pasal: {predicted_pasal} | Referensi Putusan: {predicted_text[:100]}..."
    
    top_5_ids = [c["case_id"] for c in top_k]
    
    return predicted_solution, top_5_ids

def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    df, vectorizer, tfidf_matrix, case_solutions = load_data_and_vectorize()
    
    with open(QUERIES_JSON, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    print("Mulai melakukan prediksi solusi (Case Solution Reuse)...")
    
    predictions = []
    for q in queries:
        query_id = q["query_id"]
        query_text = q["query_text"]
        
        # Lakukan prediksi
        predicted_solution, top_5_ids = predict_outcome(
            query_text, vectorizer, tfidf_matrix, df, case_solutions
        )
        
        # Simpan hasil
        predictions.append({
            "query_id": query_id,
            "predicted_solution": predicted_solution,
            "top_5_case_ids": ", ".join(top_5_ids)
        })
        
        print(f"[{query_id}] Prediksi selesai.")
        
    # Simpan ke CSV
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8")
    
    print(f"\nPrediksi selesai! Data disimpan di: {PREDICTIONS_CSV}")
    print("\nHasil Prediksi:")
    print(pred_df)

if __name__ == "__main__":
    main()
