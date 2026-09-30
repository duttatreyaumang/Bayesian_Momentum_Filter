import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import beta

# 1. Load the 2024 Formation Data
df = pl.read_parquet(r"Data/market_data_2024.parquet")

# 2. Extract Cross-Sectional Metrics
summary = df.group_by("Ticker").agg([
    (pl.col("Daily_Return") > 0).sum().alias("Pos_Days"),
    pl.col("Daily_Return").drop_nulls().count().alias("Total_Days"),
    
    # Conditional magnitudes (mean of positive days and negative days)
    pl.col("Daily_Return").filter(pl.col("Daily_Return") > 0).mean().alias("Mean_Pos_Ret"),
    pl.col("Daily_Return").filter(pl.col("Daily_Return") < 0).mean().alias("Mean_Neg_Ret")
]).filter(pl.col("Total_Days") > 0)

# Calculate the probability of an Up-Day per ticker
summary = summary.with_columns(
    (pl.col("Pos_Days") / pl.col("Total_Days")).alias("P_Up")
)

# 3. Fit the Beta Distribution via MLE
p_up_array = summary["P_Up"].drop_nulls().to_numpy()
# Fix location and scale to [0, 1] as probabilities are bounded
alpha_0, beta_0, _, _ = beta.fit(p_up_array, floc=0, fscale=1)

print(f"Empirical Prior: alpha_0 = {alpha_0:.4f}, beta_0 = {beta_0:.4f}")

# ==========================================
# 4. Generate Presentation Visualizations
# ==========================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: The Direction Distribution (Beta Prior)
sns.histplot(p_up_array, bins=50, stat="density", alpha=0.6, color="royalblue", ax=axes[0], label="Empirical Data")
x = np.linspace(0, 1, 100)
axes[0].plot(x, beta.pdf(x, alpha_0, beta_0), 'r--', lw=2, label=f'Beta Fit ($\\alpha={alpha_0:.2f}, \\beta={beta_0:.2f}$)')
axes[0].set_title("Distribution of Up-Day Probabilities (2024)", fontsize=14)
axes[0].set_xlabel("Probability of Positive Daily Return", fontsize=12)
axes[0].set_ylabel("Density", fontsize=12)
axes[0].legend()

# Plot 2: The Magnitude Distribution (Conditional Expectations)
sns.kdeplot(summary["Mean_Pos_Ret"].drop_nulls(), fill=True, color="seagreen", ax=axes[1], label="Positive Days ($E[R_A]$)")
sns.kdeplot(summary["Mean_Neg_Ret"].drop_nulls(), fill=True, color="indianred", ax=axes[1], label="Negative Days ($E[R_B]$)")
axes[1].set_title("Conditional Return Magnitudes (2024)", fontsize=14)
axes[1].set_xlabel("Mean Daily Return", fontsize=12)
axes[1].set_ylabel("Density", fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.show()
