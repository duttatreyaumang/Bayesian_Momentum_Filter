# Bayesian_Momentum_Estimation
A recursive Bayesian state filter transforming Frog-in-the-Pan (FIP) momentum into dynamic expected returns for dual-regime alpha generation.

# Recursive Bayesian Estimation for Cross-Sectional Momentum States

A quantitative research and algorithmic trading framework that transforms static, backward-looking Information Discreteness (ID)—derived from the Frog-in-the-Pan (FIP) behavioral momentum hypothesis—into a dynamic, recursive Bayesian state filter.

By continuously updating Beta-Bernoulli conjugate priors with an exponential decay factor, this framework dynamically models daily trend persistence and return conditional expectations to generate forward-looking expected returns ($E[R_{t+1}]$) across cross-sectional equities.

---

## Theoretical Framework

### The Frog-in-the-Pan (FIP) Anomaly
* **Behavioral Premise**: Due to limited cognitive bandwidth, investors price large, sudden information shocks immediately but systematically underreact to small, continuous, and incremental information arrivals.
* **Information Discreteness (ID)**: Traditional FIP models quantify this dynamic by measuring how evenly return paths are distributed:
  $$ID = \text{sgn}(PRET) \times [\%neg - \%pos]$$
  where $PRET$ represents past cumulative formation returns, and $\%pos$ and $\%neg$ are the percentages of positive and negative return days over the formation period. Continuous price paths produce persistent momentum, whereas discrete price paths prompt swift mean-reversion.

### Limitations of Static FIP
1. **Rigid Lookback Windows**: Relies on static 252-day windows, weighting day 1 identically to day 252 and ignoring recent structural fatigue.
2. **Cold-Start Bias**: Assuming uniform or arbitrary starting distributions causes early-window instability.
3. **Static Return Payoffs**: Ignores shifting volatility regimes by assuming historical payout sizes remain fixed.

### The Bayesian State-Space Upgrade
Instead of treating trend persistence as a static accounting ratio, this framework maps daily binary outcomes (up-days vs. down-days) into a dynamic probability distribution updated sequentially as each trading day closes[cite: 1]:

$$\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$$

$$\text{Beta}(\alpha_t, \beta_t) \xrightarrow{\text{Decay } \lambda} \text{Beta}(\lambda \alpha_t, \lambda \beta_t) + (x_{t+1}, 1 - x_{t+1})$$

1. **Directional Prior (Beta-Bernoulli Conjugate)**: Calibrated via Maximum Likelihood Estimation (MLE) across cross-sectional market data to establish empirical initial parameters ($\alpha_0 = 128.5, \beta_0 = 117.7$)[cite: 1].
2. **Dynamic Likelihood Updating**: Incorporates conditional payoff expectations for positive days ($R_{A,t}$) and negative days ($R_{B,t}$) for each asset[cite: 1].
3. **Forward-Looking Expected Return**:
   $$E[R_{t+1}] = P(\text{Up})_t \cdot R_{A,t} + P(\text{Down})_t \cdot R_{B,t}$$
   where $P(\text{Up})_t = \frac{\alpha_t}{\alpha_t + \beta_t}$[cite: 1].

---

## Memory Tuning & Dual Alpha Generation

The exponential decay factor ($\lambda$) dictates the filter's effective lookback memory half-life[cite: 1]:

$$\text{Half-Life} \approx \frac{\ln(0.5)}{\ln(\lambda)}$$

Tuning $\lambda$ allows the exact same underlying recursive filter to extract two orthogonal alpha streams[cite: 1]:

* **Signal 1: Medium-Term Momentum ($\lambda = 0.99$ to $0.998$)**:
  * **Half-life**: ~69 to 346 trading days[cite: 1].
  * **Mechanism**: Filters out high-frequency noise and isolates the structural 12-month underreaction premium[cite: 1].
  * **Execution**: Long continuous winners, short continuous losers[cite: 1].
* **Signal 2: Short-Term Mean Reversion ($\lambda = 0.98$)**:
  * **Half-life**: ~34 trading days[cite: 1].
  * **Mechanism**: Aggressively tracks immediate price overextension and trend exhaustion[cite: 1].
  * **Execution**: Inverted sort—long overextended losers, short overextended winners[cite: 1].

---

## Out-of-Sample Performance (2025)

Evaluated out-of-sample across 2025 daily adjusted pricing data, using 2024 exclusively for in-sample Empirical Bayes prior estimation[cite: 1].

| Strategy / Calibration | Decay ($\lambda$) | Execution Logic | Ann. Return | Ann. Volatility | Sharpe Ratio | Max Drawdown |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Vanilla FIP Benchmark** | Static (1Y) | Long Continuous Winners / Short Losers | 8.97% | 9.49% | 0.94 | -6.61% |
| **Fast Decay (Misaligned Momentum)** | 0.98 | Long Overextended Winners / Short Losers | -10.40% | 13.02% | -0.80 | -11.99% |
| **Signal 1: Momentum Calibration** | 0.99 | Long Continuous Winners / Short Losers | 5.01% | 11.36% | 0.44 | -5.25% |
| **Signal 2: Reversal Calibration** | 0.98 | Long Overextended Losers / Short Winners | 7.53% | 12.76% | 0.59 | -7.96% |

*(All backtest metrics reflect dollar-neutral, cross-sectional long/short portfolios[cite: 1].)*
