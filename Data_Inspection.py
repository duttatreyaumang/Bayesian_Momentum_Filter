import polars as pl

# 1. Load the Parquet file
df = pl.read_parquet(r"Data\market_data.parquet")

# 2. Group by ticker and aggregate the date ranges
ticker_summary = df.group_by("Ticker").agg([
    pl.col("Date").min().alias("Start_Date"),
    pl.col("Date").max().alias("End_Date"),
    pl.col("Date").count().alias("Total_Trading_Days")
])

# 3. Sort by total days to easily see which tickers have the least/most data
ticker_summary = ticker_summary.sort("Total_Trading_Days")

print("Summary of Date Ranges per Ticker:")
print(ticker_summary.head(10)) # Shows the tickers with the shortest history
print(ticker_summary.tail(10)) # Shows the tickers with the longest history

# 4. Check for uniformity
unique_starts = ticker_summary["Start_Date"].n_unique()
unique_ends = ticker_summary["End_Date"].n_unique()
unique_lengths = ticker_summary["Total_Trading_Days"].n_unique()

print("\n--- Uniformity Check ---")
print(f"Different Start Dates found: {unique_starts}")
print(f"Different End Dates found: {unique_ends}")
print(f"Different Trading Day Lengths found: {unique_lengths}")
