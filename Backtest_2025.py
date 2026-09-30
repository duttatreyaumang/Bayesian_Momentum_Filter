import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# ==========================================
# 1. FORMATION PERIOD WITH DEPENDENT SORT (2024)
# ==========================================
df_2024 = pl.read_parquet(r"Data/market_data_2024.parquet")

# Calculate ID and identify past 12-month metrics
fip_2024 = df_2024.group_by("Ticker").agg([
    ((pl.col("Adj_Close").last() / pl.col("Adj_Close").first()) - 1).alias("R_cum"),
    (pl.col("Daily_Return") > 0).sum().alias("Pos_Days"),
    (pl.col("Daily_Return") < 0).sum().alias("Neg_Days"),
    pl.col("Daily_Return").drop_nulls().count().alias("Total_Days")
]).with_columns(
    (
        pl.col("R_cum").sign() * 
        ((pl.col("Neg_Days") / pl.col("Total_Days")) - (pl.col("Pos_Days") / pl.col("Total_Days")))
    ).alias("ID")
)

# Step 1: Isolate the Return Quintiles first
fip_2024 = fip_2024.filter(pl.col("Total_Days") >= 200).with_columns(
    pl.col("R_cum").qcut(5, labels=["Ret_Q1_Losers", "Ret_Q2", "Ret_Q3", "Ret_Q4", "Ret_Q5_Winners"]).alias("Ret_Quintile")
)

winners = fip_2024.filter(pl.col("Ret_Quintile") == "Ret_Q5_Winners")
losers = fip_2024.filter(pl.col("Ret_Quintile") == "Ret_Q1_Losers")

# Step 2: Dependent Sort - Take the most continuous half (lowest ID) WITHIN each group
long_targets = winners.filter(
    pl.col("ID") <= pl.col("ID").median()
).select(pl.col("Ticker"), pl.lit("Q1").alias("Quintile"))

short_targets = losers.filter(
    pl.col("ID") <= pl.col("ID").median()
).select(pl.col("Ticker"), pl.lit("Q5").alias("Quintile"))

portfolio_targets = pl.concat([long_targets, short_targets])

print("Long Leg (Continuous Winners) count:", long_targets.height)
print("Short Leg (Continuous Losers) count:", short_targets.height)

# ==========================================
# 2. BACKTESTING PERIOD (2025)
# ==========================================
df_2025 = pl.read_parquet(r"Data/market_data_2025.parquet")

# Calculate 2025 Daily Returns on the fly before joining
df_2025 = df_2025.sort(["Ticker", "Date"]).with_columns(
    (pl.col("Adj_Close") / pl.col("Adj_Close").shift(1) - 1)
    .over("Ticker")
    .alias("Daily_Return")
)

# Join the 2024 portfolio assignments to the 2025 daily price action
backtest = df_2025.join(portfolio_targets, on="Ticker", how="inner")

# Calculate Equal-Weighted Daily Returns for Q1 and Q5
daily_returns = backtest.group_by(["Date", "Quintile"]).agg(
    pl.col("Daily_Return").mean().alias("Leg_Return")
)

# Pivot to place Q1 and Q5 returns side-by-side per Date
daily_spread = daily_returns.pivot(
    index="Date",
    on="Quintile",
    values="Leg_Return"
).sort("Date")

# Handle any initial null returns and calculate strategy return
daily_spread = daily_spread.with_columns([
    pl.col("Q1").fill_null(0.0),
    pl.col("Q5").fill_null(0.0)
]).with_columns(
    ((pl.col("Q1") - pl.col("Q5")) / 2).alias("Strategy_Return")
)

# Compute Cumulative Returns for the legs and the combined strategy
daily_spread = daily_spread.with_columns([
    (1 + pl.col("Q1")).cum_prod().alias("Long_Leg_Cum"),
    (1 + pl.col("Q5")).cum_prod().alias("Short_Leg_Cum"),
    (1 + pl.col("Strategy_Return")).cum_prod().alias("Strategy_Cum")
])

# Performance & Risk Analytics
strategy_returns = daily_spread["Strategy_Return"].drop_nulls()

ann_return = daily_spread["Strategy_Cum"].last() - 1
ann_vol = strategy_returns.std() * (252 ** 0.5)
sharpe = ann_return / ann_vol if ann_vol > 0 else 0

daily_spread = daily_spread.with_columns(
    pl.col("Strategy_Cum").cum_max().alias("Peak")
).with_columns(
    ((pl.col("Strategy_Cum") - pl.col("Peak")) / pl.col("Peak")).alias("Drawdown")
)
max_drawdown = daily_spread["Drawdown"].min()
fitness = sharpe * (abs(ann_return) ** 0.5)

print("\n--- Strategy Performance Metrics ---")
print(f"Annualized Return: {ann_return:.2%}")
print(f"Annualized Volatility (Risk): {ann_vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Estimated Fitness: {fitness:.2f}")

# ==========================================
# 3. PRESENTATION VISUALIZATIONS (FIXED DATE PARSING)
# ==========================================
# Slice '2025-01-02' out of '2025-01-02 00:00:00-05:00' and convert to Date
if daily_spread["Date"].dtype == pl.Utf8:
    daily_spread = daily_spread.with_columns(
        pl.col("Date").str.slice(0, 10).str.to_date("%Y-%m-%d")
    )
elif daily_spread["Date"].dtype in [pl.Datetime, pl.Date]:
    daily_spread = daily_spread.with_columns(pl.col("Date").cast(pl.Date))

dates = daily_spread["Date"].to_list()

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Panel 1: Cumulative Multipliers
axes[0].plot(dates, daily_spread["Long_Leg_Cum"].to_numpy(), label="Long Leg (Continuous Winners)", color="mediumseagreen", lw=2)
axes[0].plot(dates, daily_spread["Short_Leg_Cum"].to_numpy(), label="Short Leg (Continuous Losers)", color="indianred", lw=2, linestyle="--")
axes[0].plot(dates, daily_spread["Strategy_Cum"].to_numpy(), label="Dollar-Neutral Strategy (L/S)", color="royalblue", lw=2.5)

axes[0].axhline(1.0, color="gray", linestyle=":", alpha=0.7)
axes[0].set_title("Vanilla FIP Replication: Out-of-Sample Performance (2025)", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Cumulative Multiplier ($)", fontsize=12)
axes[0].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

# Panel 2: Drawdown Profile
axes[1].fill_between(dates, daily_spread["Drawdown"].to_numpy() * 100, 0, color="crimson", alpha=0.4, label="Strategy Drawdown (%)")
axes[1].plot(dates, daily_spread["Drawdown"].to_numpy() * 100, color="crimson", lw=1)
axes[1].axhline(0, color="black", linestyle="--", lw=0.8)
axes[1].set_title("Strategy Drawdown Profile", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Drawdown (%)", fontsize=11)
axes[1].set_xlabel("Date", fontsize=12)
axes[1].set_ylim(bottom=min(daily_spread["Drawdown"].min() * 100 * 1.2, -10), top=1)

# Format X-Axis with clean Monthly ticks
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

for ax in axes:
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.grid(False, which='minor')

fig.autofmt_xdate(rotation=30, ha='right')

plt.tight_layout()
plt.savefig("Vanilla_FIP_2025_Replication.png", dpi=300)
plt.show()
