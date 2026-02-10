import polars as pl
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import jieba

def stage_c_tfidf_pca():
    base_dir = Path(__file__).resolve().parent
    monthly_path = base_dir / "monthly_corpus.parquet"
    
    df = pl.read_parquet(monthly_path).to_pandas()
    
    # Chinese tokenization for better TF-IDF
    print("Tokenizing monthly docs...")
    df['tokens'] = df['text_month'].apply(lambda x: " ".join(jieba.cut(x)))
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words=None) # Stop words usually need a list
    tfidf_matrix = vectorizer.fit_transform(df['tokens'])
    
    print("Running PCA...")
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(tfidf_matrix.toarray())
    
    for i in range(3):
        df[f'tfidf_pca{i+1}'] = pca_features[:, i]
    
    # Get top terms for PCA1
    feature_names = vectorizer.get_feature_names_out()
    pca1_components = pca.components_[0]
    top_indices = pca1_components.argsort()[-10:][::-1]
    top_terms = [feature_names[i] for i in top_indices]
    print(f"Top terms for PCA1: {top_terms}")
    
    output_df = df[['month', 'tfidf_pca1', 'tfidf_pca2', 'tfidf_pca3']]
    output_csv = base_dir / "policy_monthly_pca.csv"
    output_df.to_csv(output_csv, index=False)
    print(f"Stage C Complete: Saved {output_csv}")

if __name__ == "__main__":
    # Check if jieba is installed
    try:
        import jieba
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "jieba"])
    
    stage_c_tfidf_pca()
