import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import seaborn as sns

# ==========================================
# 1. LOAD DATA & FIND REAL TRENDING STOCKS
# ==========================================
df_2025 = pl.read_parquet(r"Data/Market_Data_2025.parquet")

# Calculate daily returns
df_2025 = df_2025.sort(["Ticker", "Date"]).with_columns(
    (pl.col("Adj_Close") / pl.col("Adj_Close").shift(1) - 1).over("Ticker").alias("Daily_Return")
).drop_nulls("Daily_Return")

# Get the first 5 trading dates of 2025
dates = df_2025["Date"].unique().sort().to_list()
first_5_dates = dates[:5]

# Filter data for just the first week
df_first_week = df_2025.filter(pl.col("Date").is_in(first_5_dates))

# Find the most continuous winner and loser in this specific week
stats_first_week = df_first_week.group_by("Ticker").agg([
    (pl.col("Daily_Return") > 0).sum().alias("Up_Days"),
    (pl.col("Daily_Return") < 0).sum().alias("Down_Days")
])

# Extract the actual ticker symbols dynamically
winner_ticker = stats_first_week.sort("Up_Days", descending=True)["Ticker"][0]
loser_ticker = stats_first_week.sort("Down_Days", descending=True)["Ticker"][0]

print(f"Tracking Real Winner: {winner_ticker}")
print(f"Tracking Real Loser: {loser_ticker}")

# ==========================================
# 2. TRACK REAL BAYESIAN STATES 
# ==========================================
alpha_0 = 128.55
beta_0 = 117.66
decay = 0.98

def track_states(ticker, df, dates_list):
    a_states = [alpha_0]
    b_states = [beta_0]
    ticker_data = df.filter(pl.col("Ticker") == ticker)
    
    for d in dates_list:
        ret_series = ticker_data.filter(pl.col("Date") == d)["Daily_Return"]
        if len(ret_series) > 0:
            ret = ret_series[0]
            # Exact logic from your execution loop
            new_a = a_states[-1] * decay + (1 if ret > 0 else 0)
            new_b = b_states[-1] * decay + (1 if ret < 0 else 0)
        else:
            new_a = a_states[-1] * decay
            new_b = b_states[-1] * decay
            
        a_states.append(new_a)
        b_states.append(new_b)
        
    return a_states, b_states

a_win, b_win = track_states(winner_ticker, df_first_week, first_5_dates)
a_los, b_los = track_states(loser_ticker, df_first_week, first_5_dates)

# ==========================================
# 3. PRESENTATION VISUALIZATION
# ==========================================
# X-axis for plotting probabilities (zoom in around the mean)
x = np.linspace(0.35, 0.70, 500)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# --- PANEL 1: THE WINNING STOCK ---
axes[0].plot(x, beta.pdf(x, a_win[0], b_win[0]), label='Prior (Dec 31, 2024)', color='gray', lw=2, linestyle=':')
axes[0].fill_between(x, beta.pdf(x, a_win[0], b_win[0]), alpha=0.1, color='gray')

axes[0].plot(x, beta.pdf(x, a_win[1], b_win[1]), label='Posterior (Day 1)', color='mediumseagreen', lw=2, alpha=0.7)
axes[0].plot(x, beta.pdf(x, a_win[5], b_win[5]), label='Posterior (End of Week 1)', color='darkgreen', lw=3)
axes[0].fill_between(x, beta.pdf(x, a_win[5], b_win[5]), alpha=0.2, color='darkgreen')

axes[0].set_title(f"Real Posterior Evolution: {winner_ticker}", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Probability of an Up-Day", fontsize=12)
axes[0].set_ylabel("Density", fontsize=12)
axes[0].legend(loc="upper left")

# --- PANEL 2: THE LOSING STOCK ---
axes[1].plot(x, beta.pdf(x, a_los[0], b_los[0]), label='Prior (Dec 31, 2024)', color='gray', lw=2, linestyle=':')
axes[1].fill_between(x, beta.pdf(x, a_los[0], b_los[0]), alpha=0.1, color='gray')

axes[1].plot(x, beta.pdf(x, a_los[1], b_los[1]), label='Posterior (Day 1)', color='lightcoral', lw=2, alpha=0.7)
axes[1].plot(x, beta.pdf(x, a_los[5], b_los[5]), label='Posterior (End of Week 1)', color='darkred', lw=3)
axes[1].fill_between(x, beta.pdf(x, a_los[5], b_los[5]), alpha=0.2, color='darkred')

axes[1].set_title(f"Real Posterior Evolution: {loser_ticker}", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Probability of an Up-Day", fontsize=12)
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig("Real_Posterior_Evolution.png", dpi=300)
plt.show()
