import polars as pl

# 1. Load the 2024 data
df_2024 = pl.read_parquet(r"Data/market_data_2024.parquet")

# 2. Sort by Ticker and Date to ensure accurate return calculation
df_2024 = df_2024.sort(["Ticker", "Date"])

# 3. Add the Daily_Return column
df_2024 = df_2024.with_columns(
    (pl.col("Adj_Close") / pl.col("Adj_Close").shift(1) - 1)
    .over("Ticker")
    .alias("Daily_Return")
)

# 4. Overwrite the parquet file
df_2024.write_parquet(r"Data/market_data_2024.parquet", compression="snappy")

print("Successfully appended the 'Daily_Return' column.")
print(df_2024.head())
