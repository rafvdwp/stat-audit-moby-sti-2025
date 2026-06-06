import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import random
import math

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

class BloomFilter:
    def __init__(self, k, m):
        self.k = k
        self.m = m
        self.bit_array = [0] * m
        self.n_items = 0

    def _hash(self, item, seed):
        key = f"{seed}:{item}"
        return int(hashlib.sha256(key.encode()).hexdigest(), 16) % self.m

    def add(self, item):
        for seed in range(self.k):
            self.bit_array[self._hash(item, seed)] = 1
        self.n_items += 1

    def contains(self, item):
        return all(self.bit_array[self._hash(item, seed)] == 1 for seed in range(self.k))

    def theoretical_fpr(self, n):
        return round((1 - (1 - 1/self.m)**n)**self.k, 8)

bf = BloomFilter(k=5, m=5000)
for url in df['url'].tolist():
    bf.add(str(url))
print(f'Total issue dimasukkan: {bf.n_items}')
print(f'FPR teoritis: {bf.theoretical_fpr(bf.n_items):.6f}')

test_fake = ['https://github.com/moby/moby/issues/99999',
             'https://github.com/moby/moby/issues/00001']
for url in test_fake:
    print(f'{url} → {bf.contains(url)}')

df_mcmc = df_closed[df_closed['age_days'] > 0].copy()
df_mcmc['comments'] = pd.to_numeric(df_mcmc['comments'], errors='coerce').fillna(0)
items = [{'name': f"issue_{r['number']}", 'weight': float(r['age_days']), 'value': float(r['comments'])}
         for _, r in df_mcmc.iterrows()]

capacity = 120
n_iter = 100000
random.seed(42)
n = len(items)
current = [0] * n
best_state = current[:]
best_value = 0
T = 1.0
accepted = 0

for _ in range(n_iter):
    proposal = current[:]
    proposal[random.randint(0, n-1)] ^= 1
    w = sum(items[i]['weight'] for i in range(n) if proposal[i])
    if w > capacity:
        continue
    v_cur = sum(items[i]['value'] for i in range(n) if current[i])
    v_pro = sum(items[i]['value'] for i in range(n) if proposal[i])
    if v_pro >= v_cur or random.random() < math.exp((v_pro - v_cur) / T):
        current = proposal
        accepted += 1
    if sum(items[i]['value'] for i in range(n) if current[i]) > best_value:
        best_state = current[:]
        best_value = sum(items[i]['value'] for i in range(n) if best_state[i])
    T = max(0.01, T * 0.99995)

selected = [items[i]['name'] for i in range(n) if best_state[i] == 1]
total_w = sum(items[i]['weight'] for i in range(n) if best_state[i])
print(f'Issue terpilih : {len(selected)}')
print(f'Total nilai    : {best_value:.0f} komentar')
print(f'Total bobot    : {total_w:.0f} / {capacity} hari')
print(f'Acceptance rate: {accepted/n_iter:.4f}')
