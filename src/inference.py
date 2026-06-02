"""
inference.py
============
Modul Confidence Interval & Credible Interval
Member C: Inference Analyst | Repository: moby/moby (Docker Engine)

Tugas Akhir Kelompok — Statistika dan Probabilitas STI 2025
Semua formula bersumber dari:
  Tsun, Probability & Statistics with Applications to Computing, 2020.

Cara pakai di notebook:
    import sys
    sys.path.append("../src")
    from inference import ci_bernoulli, ci_poisson, credible_interval

Fungsi yang tersedia:
    1. confidence_interval()   — CI umum (rumus inti)
    2. ci_bernoulli()          — CI untuk proporsi Bernoulli (RQ1)
    3. ci_poisson()            — CI untuk parameter Poisson (RQ2)
    4. credible_interval()     — Credible Interval Bayesian dari Beta posterior

⚠️  LARANGAN KERAS yang harus dipatuhi di seluruh repo:
    - Jangan tulis "95% probability θ ada di sini" untuk CI Frequentist
    - Gunakan wording: "95% dari interval yang terbentuk akan mengandung θ"
"""

import numpy as np
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIDENCE INTERVAL — RUMUS INTI (UMUM)
# ══════════════════════════════════════════════════════════════════════════════

def confidence_interval(
    theta_hat: float,
    sigma: float,
    n: int,
    confidence: float = 0.95
) -> dict:
    """
    Membangun Confidence Interval umum berbasis Central Limit Theorem.

    Formula: CI = [θ̂ − z_{1−α/2} · σ/√n ,  θ̂ + z_{1−α/2} · σ/√n]
    Referensi: Tsun (2020), p. 300

    ✅  INTERPRETASI FREQUENTIST YANG BENAR:
        "Jika prosedur pengambilan sampel dan pembangunan interval ini
        diulang berkali-kali dengan ukuran n yang sama, maka [confidence×100]%
        dari interval yang terbentuk akan mengandung nilai θ sesungguhnya
        yang bersifat tetap (konstanta) dan tidak kita ketahui."

    ❌  INTERPRETASI YANG SALAH — JANGAN tulis ini di laporan:
        "Ada [confidence×100]% probabilitas bahwa θ berada di interval ini."
        → Salah karena θ adalah KONSTANTA dalam pandangan frequentist,
          bukan variabel acak yang memiliki distribusi.

    Parameters
    ----------
    theta_hat  : float — estimasi titik MLE (mis. k/n untuk Bernoulli)
    sigma      : float — standar deviasi populasi atau estimasinya
    n          : int   — ukuran sampel
    confidence : float — tingkat kepercayaan (default 0.95 = 95%)

    Returns
    -------
    dict dengan kunci:
        lower           — batas bawah interval
        upper           — batas atas interval
        margin_of_error — setengah lebar interval (z · σ/√n)
        z_critical      — nilai z kritis yang digunakan
        confidence      — tingkat kepercayaan
        theta_hat       — estimasi titik (di-echo kembali)

    Contoh (moby/moby RQ1):
        ci = confidence_interval(theta_hat=0.8405, sigma=0.366, n=884)
        # → {'lower': 0.8164, 'upper': 0.8646, 'margin_of_error': 0.0241, ...}
    """
    # Langkah 1: Hitung alpha (tingkat kesalahan) dari tingkat kepercayaan
    alpha_level = 1 - confidence

    # Langkah 2: Cari nilai z kritis (z_{1−α/2}) dari distribusi normal
    # Untuk 95%: z = norm.ppf(0.975) ≈ 1.96
    # Untuk 99%: z = norm.ppf(0.995) ≈ 2.576
    z_critical = stats.norm.ppf(1 - alpha_level / 2)

    # Langkah 3: Hitung margin of error  =  z · σ/√n
    margin = z_critical * sigma / np.sqrt(n)

    # Langkah 4: Bangun interval
    return {
        "lower"          : float(theta_hat - margin),
        "upper"          : float(theta_hat + margin),
        "margin_of_error": float(margin),
        "z_critical"     : float(z_critical),
        "confidence"     : confidence,
        "theta_hat"      : float(theta_hat),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. CI BERNOULLI — UNTUK RQ1 (MERGE RATE)
# ══════════════════════════════════════════════════════════════════════════════

def ci_bernoulli(k: int, n: int, confidence: float = 0.95) -> dict:
    """
    Confidence Interval untuk proporsi Bernoulli θ (PR merge rate).

    Untuk distribusi Bernoulli:
        θ̂   = k / n                              (MLE dari Notebook 02)
        σ   = √(θ̂ · (1 − θ̂))                    (std dev Bernoulli)
        CI  = θ̂ ± z_{1−α/2} · σ / √n            (Tsun 2020, p. 300)

    Catatan: σ/√n disebut "standard error" — mengukur ketidakpastian θ̂.

    Parameters
    ----------
    k          : int   — jumlah sukses / PR yang di-merge
    n          : int   — total percobaan / total PR (closed)
    confidence : float — tingkat kepercayaan (default 0.95)

    Returns
    -------
    dict (sama seperti confidence_interval, ditambah):
        k         — jumlah sukses (di-echo)
        n         — ukuran sampel (di-echo)
        std_error — σ/√n (standard error of the mean)

    Contoh (moby/moby):
        ci = ci_bernoulli(k=743, n=884)
        # CI 95% = [0.8164, 0.8646]
    """
    if n == 0:
        raise ValueError("ci_bernoulli: n tidak boleh 0.")
    if not (0 <= k <= n):
        raise ValueError(f"ci_bernoulli: k harus 0 ≤ k ≤ n, dapat k={k}, n={n}.")

    # Estimasi titik (sama dengan MLE)
    theta_hat = k / n

    # Standar deviasi Bernoulli: σ = √(θ̂·(1−θ̂))
    sigma     = np.sqrt(theta_hat * (1 - theta_hat))

    # Gunakan fungsi inti
    result = confidence_interval(theta_hat, sigma, n, confidence)

    # Tambahkan info konteks
    result["k"]         = k
    result["n"]         = n
    result["std_error"] = float(sigma / np.sqrt(n))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. CI POISSON — UNTUK RQ2 (LAJU BUG REPORT)
# ══════════════════════════════════════════════════════════════════════════════

def ci_poisson(data: list, confidence: float = 0.95) -> dict:
    """
    Confidence Interval untuk parameter Poisson λ (laju bug/minggu).

    Untuk distribusi Poisson: Var(X) = λ, sehingga σ = √λ̂
        λ̂  = Σxᵢ / n  =  x̄                     (MLE dari Notebook 02)
        σ  = √λ̂                                  (std dev Poisson)
        CI = λ̂ ± z_{1−α/2} · √(λ̂/n)             (Tsun 2020, p. 300)

    Parameters
    ----------
    data       : list  — daftar count kejadian per minggu (bug_count)
    confidence : float — tingkat kepercayaan

    Returns
    -------
    dict (sama seperti confidence_interval, ditambah):
        n          — jumlah minggu
        lambda_hat — nilai MLE λ̂ (alias theta_hat)
        std_error  — √(λ̂/n)

    Contoh (moby/moby):
        ci = ci_poisson(bugs_before)
        # CI 95% untuk λ_baseline
    """
    data = list(data)
    n    = len(data)
    if n == 0:
        raise ValueError("ci_poisson: data tidak boleh kosong.")

    # MLE Poisson: λ̂ = rata-rata sampel
    lambda_hat = float(sum(data)) / n

    # Standar deviasi Poisson: σ = √λ̂
    sigma = np.sqrt(lambda_hat)

    result = confidence_interval(lambda_hat, sigma, n, confidence)
    result["n"]          = n
    result["lambda_hat"] = lambda_hat
    result["std_error"]  = float(sigma / np.sqrt(n))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4. CREDIBLE INTERVAL — BAYESIAN (DARI BETA POSTERIOR)
# ══════════════════════════════════════════════════════════════════════════════

def credible_interval(
    alpha_param: int,
    beta_param: int,
    confidence: float = 0.95
) -> dict:
    """
    Credible Interval Bayesian dari distribusi Beta posterior.

    Rumus: Ambil kuantil dari CDF distribusi Beta(α, β):
        lower = F_Beta^{−1}(α_sig / 2)
        upper = F_Beta^{−1}(1 − α_sig / 2)
    Referensi: Tsun (2020), p. 312

    Dengan prior Beta(1,1) (uniform) dan data k sukses & m gagal:
        α (alpha_param) = k + 1   ← WAJIB +1, bukan k saja
        β (beta_param)  = m + 1   ← WAJIB +1, bukan m saja

    ✅  INTERPRETASI BAYESIAN YANG BENAR:
        "Ada [confidence×100]% probabilitas bahwa θ sesungguhnya berada
        di interval [lower, upper]."
        → BOLEH diucapkan karena dalam Bayesian, θ diperlakukan sebagai
          variabel acak yang memiliki distribusi posterior.

    ⬛  PERBEDAAN KUNCI vs Confidence Interval Frequentist:
        - CI Frequentist: interval yang berulang akan menangkap θ tetap
        - Credible Interval: pernyataan probabilitas LANGSUNG tentang θ
          (lebih intuitif secara bahasa, tapi membutuhkan prior)

    Parameters
    ----------
    alpha_param : int   — parameter α dari Beta posterior (= k+1)
    beta_param  : int   — parameter β dari Beta posterior (= m+1)
    confidence  : float — tingkat kepercayaan Bayesian

    Returns
    -------
    dict dengan kunci:
        lower      — batas bawah credible interval
        upper      — batas atas credible interval
        width      — lebar interval (upper - lower)
        confidence — tingkat kepercayaan
        alpha      — parameter α yang digunakan
        beta       — parameter β yang digunakan
        mean       — posterior mean = α/(α+β)
        mode       — posterior mode = (α-1)/(α+β-2)

    Contoh (moby/moby):
        cred = credible_interval(alpha_param=744, beta_param=142)
        # → {'lower': 0.8149, 'upper': 0.8631, ...}
    """
    if alpha_param <= 0 or beta_param <= 0:
        raise ValueError("credible_interval: alpha dan beta harus > 0.")

    # Tingkat kesalahan Bayesian
    alpha_sig = 1 - confidence

    # Kuantil dari distribusi Beta posterior
    lower = float(stats.beta.ppf(alpha_sig / 2,       alpha_param, beta_param))
    upper = float(stats.beta.ppf(1 - alpha_sig / 2,   alpha_param, beta_param))

    # Statistik ringkas distribusi posterior
    mean = alpha_param / (alpha_param + beta_param)
    if alpha_param > 1 and beta_param > 1:
        mode = (alpha_param - 1) / (alpha_param + beta_param - 2)
    else:
        mode = None  # Distribusi tidak memiliki mode interior (kasus degenerasi)

    return {
        "lower"     : lower,
        "upper"     : upper,
        "width"     : float(upper - lower),
        "confidence": confidence,
        "alpha"     : alpha_param,
        "beta"      : beta_param,
        "mean"      : float(mean),
        "mode"      : float(mode) if mode is not None else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. SELF-TEST (jalankan: python src/inference.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  inference.py — Self-Test dengan data moby/moby")
    print("=" * 60)

    # ── Nilai dari Notebook 02 (Member B) ─────────────────────────
    K          = 743    # PR merged
    N          = 884    # Total PR closed
    M          = N - K  # PR closed without merge = 141
    ALPHA_POST = K + 1  # = 744
    BETA_POST  = M + 1  # = 142

    # ── Test 1: CI Bernoulli ───────────────────────────────────────
    ci_95 = ci_bernoulli(K, N, confidence=0.95)
    ci_99 = ci_bernoulli(K, N, confidence=0.99)
    print("\n[1] CI Bernoulli (RQ1):")
    print(f"    θ̂_MLE = {ci_95['theta_hat']:.6f}")
    print(f"    CI 95% = [{ci_95['lower']:.6f}, {ci_95['upper']:.6f}]  ±{ci_95['margin_of_error']:.6f}")
    print(f"    CI 99% = [{ci_99['lower']:.6f}, {ci_99['upper']:.6f}]  ±{ci_99['margin_of_error']:.6f}")

    # ── Test 2: CI Poisson ─────────────────────────────────────────
    bugs_sample = [4.875] * 24  # Simulasi data Poisson
    ci_p = ci_poisson(bugs_sample, confidence=0.95)
    print(f"\n[2] CI Poisson (RQ2):")
    print(f"    λ̂     = {ci_p['lambda_hat']:.4f}")
    print(f"    CI 95% = [{ci_p['lower']:.4f}, {ci_p['upper']:.4f}]")

    # ── Test 3: Credible Interval ──────────────────────────────────
    cred_95 = credible_interval(ALPHA_POST, BETA_POST, confidence=0.95)
    cred_80 = credible_interval(ALPHA_POST, BETA_POST, confidence=0.80)
    print(f"\n[3] Credible Interval Bayesian (RQ1):")
    print(f"    Posterior: Beta({ALPHA_POST}, {BETA_POST})")
    print(f"    CI 95%  = [{cred_95['lower']:.6f}, {cred_95['upper']:.6f}]")
    print(f"    CI 80%  = [{cred_80['lower']:.6f}, {cred_80['upper']:.6f}]")
    print(f"    Mean    = {cred_95['mean']:.6f}")

    # ── Verifikasi konsistensi ─────────────────────────────────────
    print("\n[✓] Verifikasi:")
    print(f"    CI 99% lebih lebar dari 95%: {ci_99['margin_of_error'] > ci_95['margin_of_error']}")
    print(f"    θ̂ ada di dalam CI 95%: {ci_95['lower'] < ci_95['theta_hat'] < ci_95['upper']}")
    print(f"    θ̂ ada di dalam Credible Interval 95%: {cred_95['lower'] < K/N < cred_95['upper']}")

    print("\n✅  Semua self-test lulus!")