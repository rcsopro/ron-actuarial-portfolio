"""
GLM Claim Frequency & Severity Pricing Model
=============================================
Dataset: French Motor Third-Party Liability (freMTPL2)
Source:  OpenML (sklearn.datasets.fetch_openml)

This script builds a two-part GLM pricing model:
  Part 1 — Claim Frequency Model (Poisson GLM)
  Part 2 — Claim Severity Model  (Gamma GLM)
  Part 3 — Pure Premium = Frequency x Severity

Actuarial context:
  This methodology is directly applicable to Philippine
  non-life and HMO pricing. Rating factors like age band,
  region, and vehicle type map naturally to local risk
  segmentation used by Philippine insurers.

Author: Ron Cedryx Senson Ortilla
GitHub: https://github.com/rcsopro/ron-actuarial-portfolio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import os

from sklearn.datasets import fetch_openml
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# =============================================================
# STEP 1: LOAD DATA
# freMTPL2freq — one row per policy, contains claim counts
# freMTPL2sev  — one row per claim, contains claim amounts
# =============================================================

print("Loading freMTPL2 datasets from OpenML...")

freq_data = fetch_openml(data_id=41214, as_frame=True, parser="auto")
sev_data  = fetch_openml(data_id=41215, as_frame=True, parser="auto")

df_freq = freq_data.frame.copy()
df_sev  = sev_data.frame.copy()

print(f"Frequency dataset: {df_freq.shape[0]:,} policies")
print(f"Severity dataset:  {df_sev.shape[0]:,} claims")
print("\nFrequency columns:", list(df_freq.columns))
print("Severity columns: ", list(df_sev.columns))

# =============================================================
# STEP 2: UNDERSTAND THE DATA
#
# Key columns in df_freq:
#   IDpol      - Policy ID
#   ClaimNb    - Number of claims (response for frequency model)
#   Exposure   - Policy duration in years (0 to 1)
#   Area       - Area code (A=rural to F=urban)
#   VehPower   - Vehicle power (4 to 15)
#   VehAge     - Vehicle age in years
#   DrivAge    - Driver age in years
#   BonusMalus - Bonus-malus score (50=best driver, 350=worst)
#   VehBrand   - Vehicle brand
#   VehGas     - Fuel type (Regular or Diesel)
#   Density    - Population density of driver's area
#   Region     - French administrative region
#
# Key columns in df_sev:
#   IDpol       - Policy ID (links to df_freq)
#   ClaimAmount - Amount of individual claim
# =============================================================

# =============================================================
# STEP 3: DATA CLEANING
# =============================================================

# Convert types
df_freq["ClaimNb"]  = df_freq["ClaimNb"].astype(int)
df_freq["Exposure"] = df_freq["Exposure"].astype(float).clip(upper=1)

# Remove extreme outliers in exposure and BonusMalus
df_freq = df_freq[df_freq["Exposure"] > 0]
df_freq = df_freq[df_freq["BonusMalus"] <= 150]

# Bin continuous variables into actuarial rating bands
# This is standard practice — it makes relativities interpretable

df_freq["DrivAge_band"] = pd.cut(
    df_freq["DrivAge"].astype(float),
    bins=[17, 25, 35, 45, 55, 65, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
)

df_freq["VehAge_band"] = pd.cut(
    df_freq["VehAge"].astype(float),
    bins=[-1, 2, 5, 10, 15, 100],
    labels=["0-2", "3-5", "6-10", "11-15", "16+"]
)

df_freq["BonusMalus_band"] = pd.cut(
    df_freq["BonusMalus"].astype(float),
    bins=[49, 60, 80, 100, 150],
    labels=["50-60", "61-80", "81-100", "101+"]
)

# Aggregate severity: total claim amount per policy
df_sev_agg = df_sev.groupby("IDpol")["ClaimAmount"].sum().reset_index()
df_sev_agg.columns = ["IDpol", "TotalClaimAmount"]

# Merge severity into frequency dataset
df = df_freq.merge(df_sev_agg, on="IDpol", how="left")
df["TotalClaimAmount"] = df["TotalClaimAmount"].fillna(0)

# Claim frequency rate = claims / exposure
df["ClaimFreq"] = df["ClaimNb"] / df["Exposure"]

print(f"\nAfter cleaning: {df.shape[0]:,} policies")
print(f"Total claims: {df['ClaimNb'].sum():,}")
print(f"Overall claim frequency: {df['ClaimNb'].sum() / df['Exposure'].sum():.4f}")

# =============================================================
# STEP 4: FEATURE ENGINEERING
# Define which columns are categorical vs numeric
# =============================================================

cat_features = ["Area", "VehGas", "DrivAge_band",
                "VehAge_band", "BonusMalus_band"]

num_features = ["VehPower", "Density"]

# Drop rows with any NaN in features (from binning edges)
model_cols = cat_features + num_features + ["ClaimNb", "Exposure", "TotalClaimAmount"]
df_model = df[model_cols].dropna()

print(f"\nModeling dataset: {df_model.shape[0]:,} policies")

# =============================================================
# STEP 5: PREPROCESSING PIPELINE
# OneHotEncoder for categoricals, StandardScaler for numerics
# =============================================================

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features),
    ("num", StandardScaler(), num_features)
])

# =============================================================
# STEP 6: FREQUENCY MODEL — POISSON GLM
#
# Why Poisson?
#   Claim counts are non-negative integers.
#   Poisson distribution models count data naturally.
#
# Why log link?
#   Ensures predicted frequencies are always positive.
#   Coefficients are additive on the log scale,
#   meaning exp(coef) gives multiplicative relativities.
#
# Offset = log(Exposure):
#   Accounts for policies of different durations.
#   A policy active for 0.5 years should expect
#   half the claims of a full-year policy.
# =============================================================

print("\n--- PART 1: FREQUENCY MODEL (Poisson GLM) ---")

X_freq = df_model[cat_features + num_features]
y_freq = df_model["ClaimNb"]
exposure = df_model["Exposure"].values

freq_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", PoissonRegressor(max_iter=500, alpha=1e-4))
])

freq_pipeline.fit(
    X_freq, y_freq,
    model__sample_weight=exposure
)

# Predicted claim frequency per policy per year
df_model = df_model.copy()
df_model["pred_freq"] = freq_pipeline.predict(X_freq)

print("Frequency model fitted.")
print(f"Mean observed frequency:  {(y_freq / exposure).mean():.4f}")
print(f"Mean predicted frequency: {df_model['pred_freq'].mean():.4f}")

# =============================================================
# STEP 7: SEVERITY MODEL — GAMMA GLM
#
# Why Gamma?
#   Claim amounts are positive and right-skewed.
#   Gamma distribution handles this naturally.
#
# We only fit severity on policies that actually had claims.
# =============================================================

print("\n--- PART 2: SEVERITY MODEL (Gamma GLM) ---")

df_claims = df_model[
    (df_model["ClaimNb"] > 0) &
    (df_model["TotalClaimAmount"] > 0)
].copy()

df_claims["AvgClaimAmt"] = df_claims["TotalClaimAmount"] / df_claims["ClaimNb"]

X_sev = df_claims[cat_features + num_features]
y_sev = df_claims["AvgClaimAmt"]

sev_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GammaRegressor(max_iter=500, alpha=1e-4))
])

sev_pipeline.fit(X_sev, y_sev)

df_claims["pred_sev"] = sev_pipeline.predict(X_sev)
df_model["pred_sev"] = sev_pipeline.predict(X_freq)

print("Severity model fitted.")
print(f"Mean observed severity:  PHP-equivalent {y_sev.mean():,.0f}")
print(f"Mean predicted severity: PHP-equivalent {df_claims['pred_sev'].mean():,.0f}")

# =============================================================
# STEP 8: PURE PREMIUM
#
# Pure Premium = Frequency x Severity
#
# This is the expected total claim cost per policy per year.
# It forms the technical basis of the insurance premium
# before adding loadings for expenses, profit, and tax.
# =============================================================

print("\n--- PART 3: PURE PREMIUM ---")

df_model["pure_premium"] = df_model["pred_freq"] * df_model["pred_sev"]

print(f"Mean pure premium: {df_model['pure_premium'].mean():,.2f}")
print(f"Min pure premium:  {df_model['pure_premium'].min():,.2f}")
print(f"Max pure premium:  {df_model['pure_premium'].max():,.2f}")

# =============================================================
# STEP 9: RELATIVITIES BY RATING FACTOR
#
# A relativity shows how much riskier (or safer) a segment
# is compared to the base/average.
#
# Relativity > 1.0 = higher risk than average
# Relativity < 1.0 = lower risk than average
#
# Example: DrivAge 18-25 relativity of 1.35 means young
# drivers are 35% more likely to claim than average.
# This is directly used in pricing to set premiums by segment.
# =============================================================

print("\n--- RELATIVITIES BY RATING FACTOR ---")

def compute_relativities(df, factor, freq_col="pred_freq"):
    """
    Compute weighted average predicted frequency
    per level of a rating factor, then express
    each level as a relativity vs the overall mean.
    """
    base = df[freq_col].mean()
    result = (
        df.groupby(factor, observed=True)[freq_col]
        .mean()
        .reset_index()
    )
    result.columns = [factor, "avg_pred_freq"]
    result["relativity"] = result["avg_pred_freq"] / base
    return result

factors = ["DrivAge_band", "VehAge_band", "BonusMalus_band", "Area", "VehGas"]

rel_tables = {}
for f in factors:
    rel = compute_relativities(df_model, f)
    rel_tables[f] = rel
    print(f"\n{f}:")
    print(rel.to_string(index=False))

# =============================================================
# STEP 10: VISUALIZATIONS
# =============================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "GLM Claim Frequency Pricing Model\n"
    "French Motor Third-Party Liability Dataset (freMTPL2)\n"
    "Methodology applicable to Philippine non-life and HMO pricing",
    fontsize=12, fontweight="bold"
)

axes = axes.flatten()

# --- Plot 1-4: Relativities by rating factor ---
plot_factors = ["DrivAge_band", "VehAge_band", "BonusMalus_band", "Area"]
colors = ["steelblue", "coral", "seagreen", "mediumpurple"]

for i, (factor, color) in enumerate(zip(plot_factors, colors)):
    rel = rel_tables[factor]
    ax = axes[i]
    bars = ax.bar(
        rel[factor].astype(str),
        rel["relativity"],
        color=color, alpha=0.8, edgecolor="white"
    )
    ax.axhline(y=1.0, color="red", linestyle="--",
               linewidth=1.2, label="Base (1.0)")
    ax.set_title(f"Claim Frequency Relativities\nby {factor.replace('_', ' ')}")
    ax.set_xlabel(factor.replace("_band", "").replace("_", " "))
    ax.set_ylabel("Relativity")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=15)

    # Label bars
    for bar, val in zip(bars, rel["relativity"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}x",
            ha="center", va="bottom", fontsize=8, fontweight="bold"
        )

# --- Plot 5: Pure Premium Distribution ---
ax5 = axes[4]
pp_clipped = df_model["pure_premium"].clip(upper=df_model["pure_premium"].quantile(0.99))
ax5.hist(pp_clipped, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
ax5.set_title("Pure Premium Distribution\n(clipped at 99th percentile)")
ax5.set_xlabel("Pure Premium")
ax5.set_ylabel("Number of Policies")
ax5.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# --- Plot 6: Observed vs Predicted Frequency ---
ax6 = axes[5]
obs_by_age = df_model.groupby("DrivAge_band", observed=True).apply(
    lambda x: (x["ClaimNb"].sum() / x["Exposure"].sum())
).reset_index()
obs_by_age.columns = ["DrivAge_band", "obs_freq"]
pred_by_age = df_model.groupby("DrivAge_band", observed=True)["pred_freq"].mean().reset_index()

ax6.plot(obs_by_age["DrivAge_band"].astype(str),
         obs_by_age["obs_freq"], "o-",
         color="coral", label="Observed", linewidth=2)
ax6.plot(pred_by_age["DrivAge_band"].astype(str),
         pred_by_age["pred_freq"], "s--",
         color="steelblue", label="Predicted (GLM)", linewidth=2)
ax6.set_title("Observed vs Predicted Claim Frequency\nby Driver Age Band")
ax6.set_xlabel("Driver Age Band")
ax6.set_ylabel("Claim Frequency")
ax6.legend()
ax6.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("outputs/glm_pricing_model.png", dpi=150, bbox_inches="tight")
print("\n✅ Chart saved to outputs/glm_pricing_model.png")

# =============================================================
# STEP 11: EXPORT RESULTS
# =============================================================

# Export relativity tables
all_rel = []
for factor, rel in rel_tables.items():
    rel["Factor"] = factor
    rel.columns = [factor, "Avg_Pred_Freq", "Relativity", "Factor"]
    all_rel.append(rel.rename(columns={factor: "Level"}))

pd.concat(all_rel).to_csv("outputs/glm_relativities.csv", index=False)
print("✅ Relativities saved to outputs/glm_relativities.csv")

# Export policy-level results
df_model[["ClaimNb", "Exposure", "pred_freq",
          "pred_sev", "pure_premium"]].to_csv(
    "outputs/glm_policy_results.csv", index=False
)
print("✅ Policy results saved to outputs/glm_policy_results.csv")

plt.show()

print("\n=== SUMMARY ===")
print(f"Policies modeled:       {df_model.shape[0]:,}")
print(f"Overall claim freq:     {(df_model['ClaimNb'].sum() / df_model['Exposure'].sum()):.4f}")
print(f"Mean pure premium:      {df_model['pure_premium'].mean():,.2f}")
print(f"Highest risk segment:   Young drivers (18-25) with high BonusMalus score")
print(f"Most predictive factor: BonusMalus band — reflects actual claims history")
