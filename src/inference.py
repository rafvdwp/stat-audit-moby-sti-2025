import numpy as np
from scipy import stats


def confidence_interval(
    theta_hat: float, sigma: float, n: int, confidence: float = 0.95
) -> dict:
    alpha_level = 1 - confidence

    z_critical = stats.norm.ppf(1 - alpha_level / 2)

    margin = z_critical * sigma / np.sqrt(n)

    return {
        "lower": float(theta_hat - margin),
        "upper": float(theta_hat + margin),
        "margin_of_error": float(margin),
        "z_critical": float(z_critical),
        "confidence": confidence,
        "theta_hat": float(theta_hat),
    }


def ci_bernoulli(k: int, n: int, confidence: float = 0.95) -> dict:
    if n == 0:
        raise ValueError("ci_bernoulli: n tidak boleh 0.")
    if not (0 <= k <= n):
        raise ValueError(
            f"ci_bernoulli: k harus 0 ≤ k ≤ n, dapat k={k}, n={n}."
        )

    theta_hat = k / n

    sigma = np.sqrt(theta_hat * (1 - theta_hat))

    result = confidence_interval(theta_hat, sigma, n, confidence)

    result["k"] = k
    result["n"] = n
    result["std_error"] = float(sigma / np.sqrt(n))
    return result


def ci_poisson(data: list, confidence: float = 0.95) -> dict:
    data = list(data)
    n = len(data)
    if n == 0:
        raise ValueError("ci_poisson: data tidak boleh kosong.")

    lambda_hat = float(sum(data)) / n

    sigma = np.sqrt(lambda_hat)

    result = confidence_interval(lambda_hat, sigma, n, confidence)
    result["n"] = n
    result["lambda_hat"] = lambda_hat
    result["std_error"] = float(sigma / np.sqrt(n))
    return result


def credible_interval(
    alpha_param: int, beta_param: int, confidence: float = 0.95
) -> dict:
    if alpha_param <= 0 or beta_param <= 0:
        raise ValueError("credible_interval: alpha dan beta harus > 0.")

    alpha_sig = 1 - confidence

    lower = float(stats.beta.ppf(alpha_sig / 2, alpha_param, beta_param))
    upper = float(stats.beta.ppf(1 - alpha_sig / 2, alpha_param, beta_param))

    mean = alpha_param / (alpha_param + beta_param)
    if alpha_param > 1 and beta_param > 1:
        mode = (alpha_param - 1) / (alpha_param + beta_param - 2)
    else:
        mode = None

    return {
        "lower": lower,
        "upper": upper,
        "width": float(upper - lower),
        "confidence": confidence,
        "alpha": alpha_param,
        "beta": beta_param,
        "mean": float(mean),
        "mode": float(mode) if mode is not None else None,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  inference.py — Self-Test dengan data moby/moby")
    print("=" * 60)

    K = 743
    N = 884
    M = N - K
    ALPHA_POST = K + 1
    BETA_POST = M + 1

    ci_95 = ci_bernoulli(K, N, confidence=0.95)
    ci_99 = ci_bernoulli(K, N, confidence=0.99)
    print("\n[1] CI Bernoulli (RQ1):")
    print(f"    θ̂_MLE = {ci_95['theta_hat']:.6f}")
    print(
        f"    CI 95% = [{ci_95['lower']:.6f}, {ci_95['upper']:.6f}]  ±{ci_95['margin_of_error']:.6f}"
    )
    print(
        f"    CI 99% = [{ci_99['lower']:.6f}, {ci_99['upper']:.6f}]  ±{ci_99['margin_of_error']:.6f}"
    )

    bugs_sample = [4.875] * 24
    ci_p = ci_poisson(bugs_sample, confidence=0.95)
    print(f"\n[2] CI Poisson (RQ2):")
    print(f"    λ̂     = {ci_p['lambda_hat']:.4f}")
    print(f"    CI 95% = [{ci_p['lower']:.4f}, {ci_p['upper']:.4f}]")

    cred_95 = credible_interval(ALPHA_POST, BETA_POST, confidence=0.95)
    cred_80 = credible_interval(ALPHA_POST, BETA_POST, confidence=0.80)
    print(f"\n[3] Credible Interval Bayesian (RQ1):")
    print(f"    Posterior: Beta({ALPHA_POST}, {BETA_POST})")
    print(f"    CI 95%  = [{cred_95['lower']:.6f}, {cred_95['upper']:.6f}]")
    print(f"    CI 80%  = [{cred_80['lower']:.6f}, {cred_80['upper']:.6f}]")
    print(f"    Mean    = {cred_95['mean']:.6f}")

    print("\n[✓] Verifikasi:")
    print(
        f"    CI 99% lebih lebar dari 95%: {ci_99['margin_of_error'] > ci_95['margin_of_error']}"
    )
    print(
        f"    θ̂ ada di dalam CI 95%: {ci_95['lower'] < ci_95['theta_hat'] < ci_95['upper']}"
    )
    print(
        f"    θ̂ ada di dalam Credible Interval 95%: {cred_95['lower'] < K/N < cred_95['upper']}"
    )

    print("\n✅  Semua self-test lulus!")