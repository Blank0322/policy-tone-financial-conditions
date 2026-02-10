import polars as pl
from pathlib import Path
import re

START_DATE = pl.date(2005, 1, 1)
END_DATE = pl.date(2023, 12, 31)

def stage_a_daily_corpus():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "new_txts.csv"
    
    lf = pl.scan_csv(str(data_path))
    
    # 1. Parsing
    date_expr = pl.col("Date").str.strptime(pl.Date, strict=False)
    
    # Noise pattern (ad)
    ad_pattern = r"更多数据关注公众号数据皮皮侠\s+http://www\.ppmandata\.cn/trade/list"
    
    daily_corpus = (
        lf.with_columns(
            date_expr.alias("date"),
            pl.col("Text").str.len_chars().alias("text_raw_len")
        )
        .filter(
            (pl.col("date") >= START_DATE) & (pl.col("date") <= END_DATE)
        )
        .with_columns(
            pl.col("date").dt.strftime("%Y-%m").alias("month"),
            pl.col("Text")
                .str.replace_all(ad_pattern, " ")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
                .alias("text_clean")
        )
        .select([
            "date", "month", "text_raw_len", "text_clean", "source"
        ])
    )
    
    output_path = base_dir / "daily_corpus.parquet"
    daily_corpus.collect().write_parquet(output_path, compression="zstd")
    print(f"Stage A Complete: Saved {output_path}")

def stage_b_monthly_corpus():
    base_dir = Path(__file__).resolve().parent
    daily_path = base_dir / "daily_corpus.parquet"
    
    df = pl.read_parquet(daily_path)
    
    monthly_corpus = (
        df.group_by("month")
        .agg([
            pl.col("text_clean").str.concat(" ").alias("text_month"),
            pl.len().alias("n_days"),
            pl.col("text_clean").str.len_chars().sum().alias("n_chars")
        ])
        .sort("month")
    )
    
    output_path = base_dir / "monthly_corpus.parquet"
    monthly_corpus.write_parquet(output_path, compression="zstd")
    print(f"Stage B Complete: Saved {output_path}")

if __name__ == "__main__":
    stage_a_daily_corpus()
    stage_b_monthly_corpus()
