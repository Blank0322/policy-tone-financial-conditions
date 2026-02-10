import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import re

EXPANSION_TERMS = ["稳增长", "扩大内需", "降准", "降息", "调控"]

def stage_c_aligned_pca():
    base_dir = Path(__file__).resolve().parent
    monthly_path = base_dir / "monthly_corpus.parquet"
    
    df = pl.read_parquet(monthly_path).to_pandas()
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=5000, analyzer='char', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['text_month'].str.slice(0, 50000))
    
    print("Running PCA...")
    pca = PCA(n_components=1)
    pca1 = pca.fit_transform(tfidf_matrix.toarray()).flatten()
    
    # Check loadings for alignment
    feature_names = vectorizer.get_feature_names_out()
    loadings = pd.Series(pca.components_[0], index=feature_names)
    top_pos = loadings.sort_values(ascending=False).head(20)
    top_neg = loadings.sort_values(ascending=True).head(20)
    
    print("Top Positive Loadings:\n", top_pos)
    print("Top Negative Loadings:\n", top_neg)
    
    # Alignment: check if expansion terms have positive or negative loadings
    score = 0
    for term in EXPANSION_TERMS:
        # Check single chars or bigrams in loadings
        for char in term:
            if char in loadings:
                score += loadings[char]
    
    if score < 0:
        print("Flipping PCA sign to align with expansion terms.")
        pca1 = -pca1
    
    df['tone_pca'] = pca1
    df[['month', 'tone_pca']].to_csv(base_dir / "policy_monthly_pca_aligned.csv", index=False)
    print("Aligned PCA saved.")

if __name__ == "__main__":
    stage_c_aligned_pca()
