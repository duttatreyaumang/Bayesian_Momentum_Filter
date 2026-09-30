import pandas as pd

df = pd.read_csv(r"C:\Users\udutt\OneDrive\Desktop\Trexquant Presentation\Data\stock_prices_daily.csv", parse_dates=["Date"])
df.to_parquet(r"Data\market_data.parquet", engine="pyarrow", compression="snappy")
