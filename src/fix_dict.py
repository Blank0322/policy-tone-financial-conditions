import polars as pl
from pathlib import Path
import re

EXPANSION_TERMS = ["稳增长", "扩大内需", "逆周期", "宏观调控", "降准", "降息", "稳投资", "稳就业", "纾困", "保市场主体", "稳外贸", "稳外资", "稳预期", "积极财政", "适度宽松"]
TIGHTENING_TERMS = ["去杠杆", "防风险", "严监管", "房地产调控", "遏制投机", "整顿", "治理乱象", "风险处置", "化解风险", "从严", "控制", "压降", "收紧"]

def fix_dict_aggregation():
    base_dir = Path(__file__).resolve().parent
    daily_path = base_dir / "daily_corpus.parquet"
    
    df = pl.read_parquet(daily_path)
    
    exp_regex = "|".join([re.escape(t) for t in EXPANSION_TERMS])
    tig_regex = "|".join([re.escape(t) for t in TIGHTENING_TERMS])
    
    monthly = (
        df.with_columns([
            pl.col("text_clean").str.count_matches(exp_regex).alias("exp_hits"),
            pl.col("text_clean").str.count_matches(tig_regex).alias("tig_hits"),
            pl.col("text_clean").str.len_chars().alias("n_chars")
        ])
        .group_by("month")
        .agg([
            pl.sum("exp_hits").alias("total_exp"),
            pl.sum("tig_hits").alias("total_tig"),
            pl.sum("n_chars").alias("total_chars")
        ])
        .with_columns(
            (pl.col("total_exp").cast(pl.Float64) - pl.col("total_tig").cast(pl.Float64)) / pl.col("total_chars").cast(pl.Float64) * 10000
        )
        .select([
            pl.col("month"),
            pl.col("total_exp").alias("tone_dict") # Temporary alias to reuse logic
        ])
        .with_columns(
            ((pl.col("tone_dict") - pl.col("tone_dict").mean()) / pl.col("tone_dict").std()).alias("tone_dict")
        )
        .sort("month")
    )
    
    # Let's redo the tone_dict calculation correctly in the expression
    monthly = (
        df.with_columns([
            pl.col("text_clean").str.count_matches(exp_regex).alias("exp_hits"),
            pl.col("text_clean").str.count_matches(tig_regex).alias("tig_hits"),
            pl.col("text_clean").str.len_chars().alias("n_chars")
        ])
        .group_by("month")
        .agg([
            pl.sum("exp_hits").cast(pl.Float64).alias("total_exp"),
            pl.sum("tig_hits").cast(pl.Float64).alias("total_tig"),
            pl.sum("n_chars").cast(pl.Float64).alias("total_chars")
        ])
        .with_columns(
            ((pl.col("total_exp") - pl.col("total_tig")) / pl.col("total_chars") * 10000).alias("tone_dict")
        )
        .select(["month", "tone_dict"])
        .sort("month")
    )
    
    monthly.write_csv(base_dir / "policy_monthly_dict_fixed.csv")
    print("Fixed dictionary aggregation saved.")

if __name__ == "__main__":
    fix_dict_aggregation()
