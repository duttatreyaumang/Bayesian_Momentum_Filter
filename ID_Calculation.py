import polars as pl

# 1. Load the 2024 formation data
df_2024 = pl.read_parquet(r"Data/market_data_2024.parquet")

# 2. Sort by Ticker and Date to ensure correct chronological sequence
df_2024 = df_2024.sort(["Ticker", "Date"])

# 3. Calculate Daily Returns
df_2024 = df_2024.with_columns(
    (pl.col("Adj_Close") / pl.col("Adj_Close").shift(1) - 1)
    .over("Ticker")
    .alias("Daily_Return")
)

# 4. Group by Ticker to calculate the 1-year metrics
fip_2024 = df_2024.group_by("Ticker").agg([
    # Cumulative return over the year
    ((pl.col("Adj_Close").last() / pl.col("Adj_Close").first()) - 1).alias("R_cum"),
    
    # Count of positive and negative days
    (pl.col("Daily_Return") > 0).sum().alias("Pos_Days"),
    (pl.col("Daily_Return") < 0).sum().alias("Neg_Days"),
    
    # Total valid trading days (excluding the first NaN return day)
    pl.col("Daily_Return").drop_nulls().count().alias("Total_Days")
])

# 5. Calculate Information Discreteness (ID)
fip_2024 = fip_2024.with_columns(
    (
        pl.col("R_cum").sign() * 
        ((pl.col("Neg_Days") / pl.col("Total_Days")) - (pl.col("Pos_Days") / pl.col("Total_Days")))
    ).alias("ID")
)

# 6. View the top results (Highly continuous momentum = Lowest/Negative ID)
fip_2024_sorted = fip_2024.sort("ID")
print(fip_2024_sorted.head(10))
