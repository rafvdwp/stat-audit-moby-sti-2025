# Statistical Health Report — moby/moby (Docker Engine)

> Audit statistik menyeluruh terhadap repository open-source `moby/moby`
> menggunakan konsep Statistika dan Probabilitas (Minggu 11–14), STI 2025.

## 🎯 Research Questions

| # | Pertanyaan | Teknik | Notebook |
|---|-----------|--------|---------|
| RQ1 | Berapa estimasi probabilitas sebuah PR di moby/moby berhasil di-merge, dan seberapa tidak pasti estimasi tersebut? | MLE Bernoulli + CI + Beta Posterior | `02` + `03` |
| RQ2 | Apakah laju bug report mingguan berubah signifikan setelah rilis Docker v24.0 (Mei 2023)? | MLE Poisson + Z-test | `02` + `04` |
| RQ3 | Berapa P(issue > 60 hari untuk ditutup) tanpa rumus analitik? | Monte Carlo | `05` |

## 👥 Tim

| Member | Nama | NIM | Peran |
|--------|------|-----|-------|
| A | Muhammad Raffy Dwiputra | 1519625040 | Data Engineer |
| B | Muhammad Yusuf Jiddan | 1519625062 | Estimation Analyst |
| C | Dhida Framudya Wiradonna | 1519625051 | Inference Analyst |
| D | Malik Alfat Muzaki | 1519625017 | Hypothesis Analyst |
| E | Audylia Aska Widyaputri | 1519625036 | Computation Analyst |

## 📁 Struktur Repository
- stat-audit-moby-sti-2025/
- → README.md
- → AI_USAGE_LOG.md
- → data/
-     → clean/
-        → commits_clean.csv
-        → issues_clean.csv
-        → pull_requests_clean.csv
-     → raw/
-        → commits_raw.csv
-        → fetch_data.py
-        → issues_raw.csv
-        → pull_requests_raw.csv
- → src/
-     → estimator.py   [Member B]
-     → inference.py   [Member C]
-     → hypothesis.py  [Member D]
-     → simulation.py  [Member E]
- → notebooks/
-     → 01_eda.ipynb
-     → 02_estimation.ipynb
-     → 03_confidence_interval.ipynb
-     → 04_hypothesis_testing.ipynb
-     → 05_simulation.ipynb
- → report/
-     → statistical_health_report.pdf
- → presentation/
-     → video_link.md
- → requirements.txt

## 🚀 Cara Menjalankan

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set GitHub token (opsional, tapi sangat disarankan)
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 3. Ambil data (jalankan sekali)
python fetch_data.py

# 4. Jalankan notebook secara urut
jupyter notebook
```

## 📊 Temuan Utama

Belum Selesai Menganalisis Semuanya

## 🔗 Sumber Data

- **Repository:** https://github.com/moby/moby
- **Tanggal pengambilan data:** 24 Mei 2026
- **Endpoint API:** GitHub REST API v3
- **Keterbatasan:** Data diambil [X] halaman x 100 item via pagination
