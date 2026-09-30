import pandas as pd

df = pd.read_parquet(r"Data\market_data.parquet", engine="pyarrow")
df1 = df[(df['Date'] > '2025-01-01') & (df['Date'] < '2026-01-01')]
print(df1.head(10), df1.tail(10))

df1.to_parquet(r"Data\market_data_2025.parquet", engine="pyarrow", compression="snappy")
