import pandas as pd
import numpy as np
from scipy.stats import norm

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("pull_requests_clean.csv")

# ==========================
# HANYA PR DENGAN HASIL FINAL
# ==========================

df = df[df["state"] == "closed"]

# ==========================
# SAMPLE STATISTICS
# ==========================

n = len(df)

merged = df["is_merged"].sum()

p_hat = merged / n

print(f"Sample Size (n): {n}")
print(f"Merged PRs: {merged}")
print(f"Estimated Merge Rate (p̂): {p_hat:.4f}")

# ==========================
# HYPOTHESIS
# ==========================

p0 = 0.75

# ==========================
# ONE-SAMPLE PROPORTION Z TEST
# ==========================

se = np.sqrt(
    p0 * (1 - p0) / n
)

z = (p_hat - p0) / se

# two-sided p-value
p_value = 2 * (1 - norm.cdf(abs(z)))

# ==========================
# DECISION
# ==========================

alpha = 0.05

if p_value < alpha:
    decision = "Reject H0"
else:
    decision = "Fail to Reject H0"

# ==========================
# OUTPUT
# ==========================

print("\n=== RQ1: Pull Request Merge Probability ===")
print(f"H0 : p = {p0}")
print(f"H1 : p ≠ {p0}")
print(f"Z Statistic : {z:.4f}")
print(f"p-value     : {p_value:.6f}")
print(f"Decision    : {decision}")

issues_df = pd.read_csv(
    "issues_clean.csv",
    parse_dates=["created_at"]
)

# ==========================
# LOAD DATA UNTUK RQ2
# ==========================

issues_df = pd.read_csv(
    "issues_clean.csv",
    parse_dates=["created_at"]
)

# ==========================
# SPLIT DATA
# ==========================

before = issues_df[
    issues_df["created_at"].dt.year == 2025
].copy()

after = issues_df[
    issues_df["created_at"].dt.year == 2026
].copy()

# ==========================
# ISSUE COUNT PER WEEK
# ==========================

before["week"] = before["created_at"].dt.to_period("W")
after["week"] = after["created_at"].dt.to_period("W")

before_counts = before.groupby("week").size()
after_counts = after.groupby("week").size()

# ==========================
# SAMPLE STATISTICS
# ==========================

x_bar1 = before_counts.mean()
x_bar2 = after_counts.mean()

sigma1 = before_counts.std(ddof=1)
sigma2 = after_counts.std(ddof=1)

n1 = len(before_counts)
n2 = len(after_counts)

# ==========================
# HYPOTHESIS
# ==========================

print("H0 : λ2025 = λ2026")
print("H1 : λ2025 ≠ λ2026")

# ==========================
# TWO SAMPLE Z TEST
# ==========================

z = (x_bar1 - x_bar2) / np.sqrt(
    (sigma1**2 / n1) +
    (sigma2**2 / n2)
)

p_value = 2 * (1 - norm.cdf(abs(z)))

alpha = 0.05

if p_value < alpha:
    decision = "Reject H0"
else:
    decision = "Fail to Reject H0"

# ==========================
# OUTPUT
# ==========================

print("\n=== RQ2 ===")
print(f"Mean 2025 : {x_bar1:.2f}")
print(f"Mean 2026 : {x_bar2:.2f}")
print(f"n2025     : {n1}")
print(f"n2026     : {n2}")
print(f"Z         : {z:.4f}")
print(f"p-value   : {p_value:.6f}")
print(f"Decision  : {decision}")