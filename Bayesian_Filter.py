import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. INITIALIZE PRIORS & CONDITIONAL MEANS
# ==========================================
# Load 2024 data to extract the expected magnitude of up/down days per ticker
df_2024 = pl.read_parquet(r"Data/Market_Data_2024.parquet")

# Calculate daily returns for 2024
df_2024 = df_2024.sort(["Ticker", "Date"]).with_columns(
    (pl.col("Adj_Close") / pl.col("Adj_Close").shift(1) - 1).over("Ticker").alias("Daily_Return")
)

# Extract empirical conditional means per stock
priors_df = df_2024.group_by("Ticker").agg([
    pl.col("Daily_Return").filter(pl.col("Daily_Return") > 0).mean().fill_null(0).alias("E_R_A"),
    pl.col("Daily_Return").filter(pl.col("Daily_Return") < 0).mean().fill_null(0).alias("E_R_B")
])

# Convert to dictionaries for ultra-fast O(1) lookup in our daily loop
E_R_A_map = dict(zip(priors_df["Ticker"], priors_df["E_R_A"]))
E_R_B_map = dict(zip(priors_df["Ticker"], priors_df["E_R_B"]))

# ==========================================
# 2. THE CORRECTED BAYESIAN FILTER
# ==========================================
# Load 2025 out-of-sample data
df_2025 = pl.read_parquet(r"Data/Market_Data_2025.parquet")
df_2025 = df_2025.sort(["Ticker", "Date"]).with_columns(
    (pl.col("Adj_Close") / pl.col("Adj_Close").shift(1) - 1).over("Ticker").alias("Daily_Return")
).drop_nulls("Daily_Return")

# Filter Parameters
ALPHA_0 = 128.55
BETA_0 = 117.66
LAMBDA_DECAY = 0.98  
MAGNITUDE_DECAY = 0.95 # Faster decay for return magnitudes to adapt to volatility shifts

# State Trackers
alpha_state = {t: ALPHA_0 for t in df_2025["Ticker"].unique()}
beta_state = {t: BETA_0 for t in df_2025["Ticker"].unique()}
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
    
    # 1. Calculate today's Expected Returns & Continuity
    for ticker in tickers:
        a = alpha_state[ticker]
        b = beta_state[ticker]
        
        prob_up = a / (a + b)
        expected_ret = (prob_up * e_ra_state[ticker]) + ((1 - prob_up) * e_rb_state[ticker])
        
        # Bayesian ID: negative means continuous trend, positive means discrete
        continuity_id = np.sign(expected_ret) * ((b - a) / (a + b))
        
        current_E_R[ticker] = expected_ret
        current_continuity[ticker] = continuity_id
    
    # 2. The Dependent Sort (Restored)
    er_values = list(current_E_R.values())
    if len(er_values) >= 10:
        # Step A: Isolate Winners and Losers based on Expected Return
        p80_er = np.percentile(er_values, 80)
        p20_er = np.percentile(er_values, 20)
        
        winners = [t for t in tickers if current_E_R[t] >= p80_er]
        losers = [t for t in tickers if current_E_R[t] <= p20_er]
        
        # Step B: Filter for Continuity WITHIN those groups (Lowest Bayesian ID)
        long_targets, short_targets = [], []
        
        if winners:
            winner_ids = [current_continuity[t] for t in winners]
            long_targets = [t for t in winners if current_continuity[t] <= np.median(winner_ids)]
            # The Reversal Execution: Buy the continuous losers, Short the continuous winners

            #short_targets = [t for t in winners if current_continuity[t] <= np.median(winner_ids)]
            
        if losers:
            loser_ids = [current_continuity[t] for t in losers]
            #short_targets = [t for t in losers if current_continuity[t] <= np.median(loser_ids)]
            long_targets = [t for t in losers if current_continuity[t] <= np.median(loser_ids)]


        # 3. Calculate Actual Portfolio Return
        long_rets = [returns[tickers.index(t)] for t in long_targets if t in tickers]
        short_rets = [returns[tickers.index(t)] for t in short_targets if t in tickers]
        
        avg_long = np.mean(long_rets) if long_rets else 0.0
        avg_short = np.mean(short_rets) if short_rets else 0.0
        
        strat_ret = (avg_long / 2) - (avg_short / 2)
        strategy_returns.append({"Date": date, "Strategy_Return": strat_ret})
    else:
        strategy_returns.append({"Date": date, "Strategy_Return": 0.0})
    
    # 4. Fully Dynamic Bayesian Update (Direction AND Likelihood)
    for ticker, ret in zip(tickers, returns):
        # Update Beta parameters (Direction)
        alpha_state[ticker] = (alpha_state[ticker] * LAMBDA_DECAY) + (1 if ret > 0 else 0)
        beta_state[ticker] = (beta_state[ticker] * LAMBDA_DECAY) + (1 if ret < 0 else 0)
        
        # Update Sample Likelihoods (Magnitudes) via Recursive Estimation
        if ret > 0:
            e_ra_state[ticker] = (e_ra_state[ticker] * MAGNITUDE_DECAY) + (ret * (1 - MAGNITUDE_DECAY))
        elif ret < 0:
            e_rb_state[ticker] = (e_rb_state[ticker] * MAGNITUDE_DECAY) + (ret * (1 - MAGNITUDE_DECAY))

# ==========================================
# 3. PERFORMANCE ANALYTICS & VISUALIZATION
# ==========================================
results_df = pl.DataFrame(strategy_returns)
results_df = results_df.with_columns(
    (1 + pl.col("Strategy_Return")).cum_prod().alias("Equity_Curve")
)

# Calculate Risk Metrics
ann_return = results_df["Equity_Curve"].last() - 1
ann_vol = results_df["Strategy_Return"].std() * np.sqrt(252)
sharpe = ann_return / ann_vol if ann_vol > 0 else 0

results_df = results_df.with_columns(
    pl.col("Equity_Curve").cum_max().alias("Peak")
).with_columns(
    ((pl.col("Equity_Curve") - pl.col("Peak")) / pl.col("Peak")).alias("Drawdown")
)
max_drawdown = results_df["Drawdown"].min()

print(f"\n--- Dynamic Bayesian Strategy Metrics ---")
print(f"Annualized Return: {ann_return:.2%}")
print(f"Annualized Volatility: {ann_vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# Plotting
sns.set_theme(style="darkgrid")
fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

# Equity Curve
axes[0].plot(results_df["Date"].to_numpy(), results_df["Equity_Curve"].to_numpy(), color='royalblue', lw=2)
axes[0].set_title("Dollar-Neutral Equity Curve (Bayesian Hysteresis Filter)", fontsize=14, fontweight='bold')
axes[0].set_ylabel("Cumulative Multiplier", fontsize=12)
axes[0].axhline(1.0, color='black', linestyle='--', alpha=0.5)

# Drawdown Curve
axes[1].fill_between(results_df["Date"].to_numpy(), results_df["Drawdown"].to_numpy(), 0, color='indianred', alpha=0.7)
axes[1].set_title("Strategy Drawdown", fontsize=12)
axes[1].set_ylabel("Percentage Drop", fontsize=12)

plt.tight_layout()
plt.show()
