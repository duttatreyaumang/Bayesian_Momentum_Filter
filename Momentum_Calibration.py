import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ==========================================
# 1. INITIALIZE PRIORS & CONDITIONAL MEANS (FIXED)
# ==========================================
df_2024 = pl.read_parquet(r"Data/Market_Data_2024.parquet")

df_2024 = df_2024.sort(["Ticker", "Date"]).with_columns(
    (pl.col("Adj_Close") / pl.col("Adj_Close").shift(1) - 1).over("Ticker").alias("Daily_Return")
)

# Extract BOTH Magnitudes and Directional Counts per stock!
priors_df = df_2024.group_by("Ticker").agg([
    pl.col("Daily_Return").filter(pl.col("Daily_Return") > 0).mean().fill_null(0).alias("E_R_A"),
    pl.col("Daily_Return").filter(pl.col("Daily_Return") < 0).mean().fill_null(0).alias("E_R_B"),
    (pl.col("Daily_Return") > 0).sum().alias("Pos_Days_24"),
    (pl.col("Daily_Return") < 0).sum().alias("Neg_Days_24")
])

E_R_A_map = dict(zip(priors_df["Ticker"], priors_df["E_R_A"]))
E_R_B_map = dict(zip(priors_df["Ticker"], priors_df["E_R_B"]))
Alpha_map = dict(zip(priors_df["Ticker"], priors_df["Pos_Days_24"]))
Beta_map = dict(zip(priors_df["Ticker"], priors_df["Neg_Days_24"]))

# ==========================================
# 2. THE CORRECTED BAYESIAN FILTER (MOMENTUM)
# ==========================================
df_2025 = pl.read_parquet(r"Data/Market_Data_2025.parquet")
df_2025 = df_2025.sort(["Ticker", "Date"]).with_columns(
    (pl.col("Adj_Close") / pl.col("Adj_Close").shift(1) - 1).over("Ticker").alias("Daily_Return")
).drop_nulls("Daily_Return")

# Slow Decay Parameters
LAMBDA_DECAY = 0.99
MAGNITUDE_DECAY = 0.995 
# Fallback global priors just in case a ticker didn't exist in 2024
GLOBAL_A, GLOBAL_B = 128.55, 117.66 

# STOCK-SPECIFIC PRIOR INITIALIZATION
alpha_state = {t: Alpha_map.get(t, GLOBAL_A) for t in df_2025["Ticker"].unique()}
beta_state = {t: Beta_map.get(t, GLOBAL_B) for t in df_2025["Ticker"].unique()}
e_ra_state = {t: E_R_A_map.get(t, 0.0) for t in df_2025["Ticker"].unique()}
e_rb_state = {t: E_R_B_map.get(t, 0.0) for t in df_2025["Ticker"].unique()}

dates = df_2025["Date"].unique().sort().to_list()
strategy_returns = []

for date in dates:
    daily_data = df_2025.filter(pl.col("Date") == date)
    tickers = daily_data["Ticker"].to_list()
    returns = daily_data["Daily_Return"].to_list()
    
    current_E_R = {}
    current_continuity = {}
    
    for ticker in tickers:
        a = alpha_state[ticker]
        b = beta_state[ticker]
        
        prob_up = a / (a + b)
        expected_ret = (prob_up * e_ra_state[ticker]) + ((1 - prob_up) * e_rb_state[ticker])
        
        continuity_id = np.sign(expected_ret) * ((b - a) / (a + b))
        
        current_E_R[ticker] = expected_ret
        current_continuity[ticker] = continuity_id
    
    er_values = list(current_E_R.values())
    if len(er_values) >= 10:
        p80_er = np.percentile(er_values, 80)
        p20_er = np.percentile(er_values, 20)
        
        winners = [t for t in tickers if current_E_R[t] >= p80_er]
        losers = [t for t in tickers if current_E_R[t] <= p20_er]
        
        long_targets, short_targets = [], []
        
        # Standard FIP Logic: Long Continuous Winners, Short Continuous Losers
        if winners:
            winner_ids = [current_continuity[t] for t in winners]
            long_targets = [t for t in winners if current_continuity[t] <= np.median(winner_ids)]
            
        if losers:
            loser_ids = [current_continuity[t] for t in losers]
            short_targets = [t for t in losers if current_continuity[t] <= np.median(loser_ids)]
            
        long_rets = [returns[tickers.index(t)] for t in long_targets if t in tickers]
        short_rets = [returns[tickers.index(t)] for t in short_targets if t in tickers]
        
        avg_long = np.mean(long_rets) if long_rets else 0.0
        avg_short = np.mean(short_rets) if short_rets else 0.0
        
        strat_ret = (avg_long / 2) - (avg_short / 2)
        
        # Track all three legs for the visualization
        strategy_returns.append({
            "Date": date, 
            "Strategy_Return": strat_ret,
            "Long_Return": avg_long,
            "Short_Return": avg_short
        })
    else:
        strategy_returns.append({
            "Date": date, 
            "Strategy_Return": 0.0,
            "Long_Return": 0.0,
            "Short_Return": 0.0
        })
    
    for ticker, ret in zip(tickers, returns):
        alpha_state[ticker] = (alpha_state[ticker] * LAMBDA_DECAY) + (1 if ret > 0 else 0)
        beta_state[ticker] = (beta_state[ticker] * LAMBDA_DECAY) + (1 if ret < 0 else 0)
        
        if ret > 0:
            e_ra_state[ticker] = (e_ra_state[ticker] * MAGNITUDE_DECAY) + (ret * (1 - MAGNITUDE_DECAY))
        elif ret < 0:
            e_rb_state[ticker] = (e_rb_state[ticker] * MAGNITUDE_DECAY) + (ret * (1 - MAGNITUDE_DECAY))

# ==========================================
# 3. PERFORMANCE ANALYTICS & VISUALIZATION
# ==========================================
results_df = pl.DataFrame(strategy_returns)

# Safely parse dates for matplotlib
if results_df["Date"].dtype == pl.Utf8:
    try:
        results_df = results_df.with_columns(pl.col("Date").str.slice(0, 10).str.to_date("%Y-%m-%d"))
    except Exception:
        results_df = results_df.with_columns(pl.col("Date").str.to_datetime().dt.date())
else:
    results_df = results_df.with_columns(pl.col("Date").cast(pl.Date))

# Calculate Cumulative Returns for all legs
results_df = results_df.with_columns([
    (1 + pl.col("Strategy_Return")).cum_prod().alias("Strategy_Cum"),
    (1 + pl.col("Long_Return")).cum_prod().alias("Long_Leg_Cum"),
    (1 + pl.col("Short_Return")).cum_prod().alias("Short_Leg_Cum")
])

ann_return = results_df["Strategy_Cum"].last() - 1
ann_vol = results_df["Strategy_Return"].std() * np.sqrt(252)
sharpe = ann_return / ann_vol if ann_vol > 0 else 0

results_df = results_df.with_columns(
    pl.col("Strategy_Cum").cum_max().alias("Peak")
).with_columns(
    ((pl.col("Strategy_Cum") - pl.col("Peak")) / pl.col("Peak")).alias("Drawdown")
)
max_drawdown = results_df["Drawdown"].min()

print(f"\n--- Dynamic Bayesian Strategy Metrics (Slow Decay - Momentum) ---")
print(f"Annualized Return: {ann_return:.2%}")
print(f"Annualized Volatility: {ann_vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# Plotting
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
plot_dates = results_df["Date"].to_list()

# Panel 1: Cumulative Multipliers
axes[0].plot(plot_dates, results_df["Long_Leg_Cum"].to_numpy(), label="Long Leg (Continuous Winners)", color="mediumseagreen", lw=2)
axes[0].plot(plot_dates, results_df["Short_Leg_Cum"].to_numpy(), label="Short Leg (Continuous Losers)", color="indianred", lw=2, linestyle="--")
axes[0].plot(plot_dates, results_df["Strategy_Cum"].to_numpy(), label="Dollar-Neutral Strategy (L/S)", color="royalblue", lw=2.5)

axes[0].axhline(1.0, color="gray", linestyle=":", alpha=0.7)
axes[0].set_title(r"Signal 1: Momentum Calibration ($\lambda=0.99$)", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Cumulative Multiplier ($)", fontsize=12)
axes[0].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

# Panel 2: Drawdown Profile
axes[1].fill_between(plot_dates, results_df["Drawdown"].to_numpy() * 100, 0, color="crimson", alpha=0.4, label="Strategy Drawdown (%)")
axes[1].plot(plot_dates, results_df["Drawdown"].to_numpy() * 100, color="crimson", lw=1)
axes[1].axhline(0, color="black", linestyle="--", lw=0.8)
axes[1].set_title("Strategy Drawdown Profile", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Drawdown (%)", fontsize=11)
axes[1].set_xlabel("Date", fontsize=12)

# Format X-Axis with clean Monthly ticks
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

for ax in axes:
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.grid(False, which='minor')

fig.autofmt_xdate(rotation=30, ha='right')

plt.tight_layout()
plt.savefig("Signal_1_Momentum_Calibration.png", dpi=300)
plt.show()
