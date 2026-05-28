import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import beta as beta_dist
from typing import Union

# ---------------------------------------------------------------------------
# 1. Maximum Likelihood Estimators
# ---------------------------------------------------------------------------

def mle_bernoulli(data: list[int]) -> dict:
   
    data = np.asarray(data, dtype=int)
    n = len(data)
    if n == 0:
        raise ValueError("mle_bernoulli: data must not be empty.")
    if not np.all((data == 0) | (data == 1)):
        raise ValueError("mle_bernoulli: all values must be 0 or 1.")

    k = int(data.sum())
    theta_hat = k / n  # Tsun (2020), p. 254
    std_error = np.sqrt(theta_hat * (1 - theta_hat) / n)

    return {
        "theta_hat": theta_hat,
        "k": k,
        "n": n,
        "std_error": std_error,
    }

def mle_poisson(data: list[float]) -> dict:
    
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n == 0:
        raise ValueError("mle_poisson: data must not be empty.")
    if np.any(data < 0):
        raise ValueError("mle_poisson: Poisson data must be non-negative.")

    lambda_hat = data.sum() / n  # Tsun (2020), p. 254
    std_error = np.sqrt(lambda_hat / n)

    return {
        "lambda_hat": lambda_hat,
        "n": n,
        "std_error": std_error,
    }

# ---------------------------------------------------------------------------
# 2. Beta Posterior
# ---------------------------------------------------------------------------

def beta_posterior(k: int, m: int) -> dict:
    
    if k < 0 or m < 0:
        raise ValueError("beta_posterior: k and m must be non-negative integers.")

    alpha = k + 1   # Tsun (2020), p. 269 — NEVER use k alone
    beta_param = m + 1  # Tsun (2020), p. 269 — NEVER use m alone

    # Mode is undefined (or 0/1 boundary) when α or β ≤ 1
    if alpha > 1 and beta_param > 1:
        mode = (alpha - 1) / (alpha + beta_param - 2)  # Tsun (2020), p. 269
    else:
        mode = None  # degenerate case

    mean = alpha / (alpha + beta_param)  # Tsun (2020), p. 269

    return {
        "alpha": alpha,
        "beta": beta_param,
        "mode": mode,
        "mean": mean,
        "k": k,
        "m": m,
    }


# ---------------------------------------------------------------------------
# 3. Log-Likelihood Functions
# ---------------------------------------------------------------------------

def log_likelihood_bernoulli(
    theta: Union[float, np.ndarray],
    k: int,
    n: int,
) -> Union[float, np.ndarray]:
    
    theta = np.asarray(theta, dtype=float)
    if np.any((theta <= 0) | (theta >= 1)):
        raise ValueError("log_likelihood_bernoulli: theta must be in (0, 1).")
    return k * np.log(theta) + (n - k) * np.log(1 - theta)


def log_likelihood_poisson(
    theta: Union[float, np.ndarray],
    data: list[float],
) -> Union[float, np.ndarray]:
    
    theta = np.asarray(theta, dtype=float)
    data = np.asarray(data, dtype=float)
    if np.any(theta <= 0):
        raise ValueError("log_likelihood_poisson: theta (lambda) must be > 0.")
    n = len(data)
    sum_x = data.sum()
    return sum_x * np.log(theta) - n * theta


# ---------------------------------------------------------------------------
# 4. Visualisation Helpers
# ---------------------------------------------------------------------------

def plot_log_likelihood_bernoulli(k: int, n: int, save_path: str = None) -> plt.Figure:
   
    theta_range = np.linspace(0.001, 0.999, 500)
    ll = log_likelihood_bernoulli(theta_range, k, n)
    theta_hat = k / n

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(theta_range, ll, color="#2563eb", linewidth=2.5, label="Log-likelihood ℓ(θ)")
    ax.axvline(theta_hat, color="#dc2626", linestyle="--", linewidth=1.8,
               label=f"MLE θ̂ = {theta_hat:.4f}")
    ax.scatter([theta_hat], [log_likelihood_bernoulli(theta_hat, k, n)],
               color="#dc2626", s=80, zorder=5)

    ax.set_xlabel("θ (Probability of Merge)", fontsize=12)
    ax.set_ylabel("ℓ(θ | k, n)", fontsize=12)
    ax.set_title(f"Bernoulli Log-Likelihood\n(k={k} successes, n={n} trials)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_log_likelihood_poisson(data: list[float], save_path: str = None) -> plt.Figure:
    
    data = np.asarray(data, dtype=float)
    result = mle_poisson(data)
    lambda_hat = result["lambda_hat"]

    # Plot range centered around MLE
    lo = max(0.01, lambda_hat * 0.3)
    hi = lambda_hat * 2.5
    lam_range = np.linspace(lo, hi, 500)
    ll = log_likelihood_poisson(lam_range, data)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(lam_range, ll, color="#059669", linewidth=2.5, label="Log-likelihood ℓ(λ)")
    ax.axvline(lambda_hat, color="#dc2626", linestyle="--", linewidth=1.8,
               label=f"MLE λ̂ = {lambda_hat:.4f}")
    ax.scatter([lambda_hat], [log_likelihood_poisson(lambda_hat, data)],
               color="#dc2626", s=80, zorder=5)

    ax.set_xlabel("λ (Poisson Rate)", fontsize=12)
    ax.set_ylabel("ℓ(λ | x)", fontsize=12)
    ax.set_title(f"Poisson Log-Likelihood\n(n={len(data)} observations, λ̂={lambda_hat:.4f})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_beta_posterior(k: int, m: int, save_path: str = None) -> plt.Figure:
    
    result = beta_posterior(k, m)
    alpha = result["alpha"]
    beta_p = result["beta"]
    mean_ = result["mean"]
    mode_ = result["mode"]

    theta_range = np.linspace(0.001, 0.999, 1000)
    pdf = beta_dist.pdf(theta_range, alpha, beta_p)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(theta_range, pdf, alpha=0.25, color="#7c3aed")
    ax.plot(theta_range, pdf, color="#7c3aed", linewidth=2.5,
            label=f"Beta(α={alpha}, β={beta_p})")
    ax.axvline(mean_, color="#2563eb", linestyle="--", linewidth=1.8,
               label=f"Posterior mean = {mean_:.4f}")
    if mode_ is not None:
        ax.axvline(mode_, color="#dc2626", linestyle=":", linewidth=1.8,
                   label=f"Posterior mode = {mode_:.4f}")

    ax.set_xlabel("θ (Probability of Merge)", fontsize=12)
    ax.set_ylabel("Posterior Density p(θ | data)", fontsize=12)
    ax.set_title(f"Beta Posterior Distribution\n(k={k} successes, m={m} failures)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5. Quick sanity-check (run: python src/estimator.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== estimator.py — self-test ===\n")

    # Bernoulli
    sample_binary = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    b = mle_bernoulli(sample_binary)
    print(f"MLE Bernoulli: {b}")

    # Poisson
    sample_counts = [3, 5, 2, 4, 6, 3, 5, 4, 3, 2]
    p = mle_poisson(sample_counts)
    print(f"MLE Poisson  : {p}")

    # Beta posterior
    bp = beta_posterior(k=7, m=3)
    print(f"Beta Posterior: {bp}")

    # Log-likelihoods (scalar check)
    ll_b = log_likelihood_bernoulli(b["theta_hat"], b["k"], b["n"])
    print(f"Log-lik Bernoulli at MLE : {ll_b:.4f}")
    ll_p = log_likelihood_poisson(p["lambda_hat"], sample_counts)
    print(f"Log-lik Poisson at MLE   : {ll_p:.4f}")

    print("\nAll checks passed ✓")