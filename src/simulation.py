import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../data/clean/issues_clean.csv')
df_closed = df[(df['state'] == 'closed') & (df['closed_at'].notna())]
age_days = df_closed['age_days'].dropna().astype(float).values

n_sim = 50000
sample_size = len(age_days)
results = np.empty(n_sim)
for i in range(n_sim):
    sample = np.random.choice(age_days, size=sample_size, replace=True)
    results[i] = np.mean(sample > 60)

estimate = results.mean()
ci_lower, ci_upper = np.percentile(results, [2.5, 97.5])
print(f'Estimated Probability: {estimate:.4f}')
print(f'95% Confidence Interval: {ci_lower:.4f} - {ci_upper:.4f}')

plt.hist(results, bins=30, edgecolor='black')
plt.axvline(estimate, color='red', linestyle='--')
plt.title('Monte Carlo Simulation Results')
plt.xlabel('Probability Issue > 60 Days')
plt.ylabel('Frequency')
plt.show()
