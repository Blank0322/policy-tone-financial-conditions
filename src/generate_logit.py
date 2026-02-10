import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

EXPANSION_TERMS = ["稳增长", "扩大内需", "逆周期", "宏观调控", "降准", "降息", "稳投资", "稳就业", "纾困", "保市场主体", "稳外贸", "稳外资", "稳预期", "积极财政", "适度宽松"]
TIGHTENING_TERMS = ["去杠杆", "防风险", "严监管", "房地产调控", "遏制投机", "整顿", "治理乱象", "风险处置", "化解风险", "从严", "控制", "压降", "收紧"]

def stage_d_logit_index():
    base_dir = Path(__file__).resolve().parent
    daily_path = base_dir / "daily_corpus.parquet"
    
    df = pl.read_parquet(daily_path).to_pandas()
    
    # 1. Generate Weak Labels
    print("Generating weak labels...")
    exp_regex = "|".join(EXPANSION_TERMS)
    tig_regex = "|".join(TIGHTENING_TERMS)
    
    df['exp_hits'] = df['text_clean'].str.count(exp_regex)
    df['tig_hits'] = df['text_clean'].str.count(tig_regex)
    
    # Label: 1 for expansion, 0 for tightening, ignore neutral
    df['label'] = np.where(df['exp_hits'] > df['tig_hits'], 1, 
                           np.where(df['tig_hits'] > df['exp_hits'], 0, -1))
    
    train_df = df[df['label'] != -1].copy()
    print(f"Training on {len(train_df)} labeled days...")
    
    # 2. Train Logistic Regression
    vectorizer = TfidfVectorizer(max_features=5000, analyzer='char', ngram_range=(1, 2))
    X = vectorizer.fit_transform(train_df['text_clean'])
    y = train_df['label']
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # 3. Predict for all days
    print("Predicting probabilities...")
    X_all = vectorizer.transform(df['text_clean'])
    df['p_expansion'] = model.predict_proba(X_all)[:, 1]
    
    # 4. Aggregate by month
    monthly_ml = df.groupby('month')['p_expansion'].mean().reset_index()
    monthly_ml.columns = ['month', 'policy_tone_ml']
    
    output_csv = base_dir / "policy_monthly_ml.csv"
    monthly_ml.to_csv(output_csv, index=False)
    print(f"Stage D Complete: Saved {output_csv}")

if __name__ == "__main__":
    stage_d_logit_index()
